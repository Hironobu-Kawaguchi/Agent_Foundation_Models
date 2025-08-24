#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_mhqa_eval_local_fixed.py

【目的】
  - NQ-Open（natural_questions_open / nq_open）等の小規模サンプルで、
    ローカルの因果言語モデル（Causal LM）を簡易評価するためのスクリプト。
  - AFM 論文（CoA: Chain-of-Agents）と互換のタグ仕様（<think>/<plan>/<tools>
    <web_search>/<crawl_page>/<observation>/<reflection>/<answer>）に対応し、
    旧 6-function 仕様（<wiki_search> 等）とも後方互換を維持します。
  - EM/F1 に加えて LLM-as-Judge での 0/1 正答評価（accuracy）を選択可能にしました。
  - 生成トレース（プロンプト末尾、出力、ツール照会/観測）を JSONL に保存可能です。

【主な修正点】
  - 🧩 CoA タグ完全対応：<tools><web_search>...</web_search></tools> を検出し
    <observation> を返すループを実装。旧 <wiki_search> も引き続き使用可。
  - 📏 評価拡張：--metric emf1 | judge（+ --judge-model）を追加。
  - 🔧 生成設定：論文で一般的な sampling 推奨値を選べるようパラメータ化。
  - 🗃️ ログ強化：トレースに tools/query/observation を保存。
  - 🪙 互換性：vanilla / sixfunc / coa の 3 プロトコルを選択可。

【依存】
  - transformers, datasets, torch, requests（web ツール使用時）

【使用例】
  # CoA プロトコル + 簡易 Wiki 検索 + LLM ジャッジ
  python run_mhqa_eval_local.py \
    --model PersonalAILab/AFM-MHQA-Agent-3B-rl \
    --dataset nq_open --split validation --n 50 --seed 42 \
    --proto coa --use-wiki \
    --metric judge --judge-model Qwen/Qwen2.5-3B-Instruct \
    --do-sample --temperature 0.8 --top-p 0.9 --top-k 20 \
    --out out_nq_rl.jsonl --save-trace --trace-max-chars 0

  # 旧 six-function 互換 + EM/F1
  python run_mhqa_eval_local.py \
    --model PersonalAILab/AFM-MHQA-Agent-3B-rl \
    --n 50 --proto sixfunc --use-wiki \
    --metric emf1 --out out_nq_emf1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

# ==========================================================
# System prompts (CoA / six-function / vanilla)
# ==========================================================
SYSTEM_COA = (
    "You can only respond using these functions: think, plan, tools, web_search, crawl_page, observation, reflection, answer.\n"
    "Rules:\n"
    "1. Start with <think>...</think> ONCE.\n"
    "2. If you need external info, emit <tools> with one or more tool calls, e.g.\n"
    "   <tools><web_search>query</web_search></tools> or <tools><crawl_page>URL</crawl_page></tools>.\n"
    "3. You will then receive <observation>...</observation> as the environment response.\n"
    "4. Use <reflection>...</reflection> to revise; you may iterate with <think>...</think>.\n"
    "5. When certain, end with <answer>SHORT FINAL ANSWER ONLY</answer>.\n"
    "6. Do NOT output any special tokens except the function tags themselves. Keep outputs minimal."
)

SYSTEM_6F = (
    "You can only respond using these 6 functions: think, plan, wiki_search, observation, reflection, answer.\n"
    "Rules:\n"
    "1. You MUST start with <think>...</think> ONCE to outline your next step.\n"
    "2. If you need external info, emit <wiki_search>query</wiki_search>. Do not fabricate observations.\n"
    "3. When a search is emitted, you will receive <observation>...</observation> back. Use it to revise.\n"
    "4. Iterate with <reflection>...</reflection> then <think>...</think>.\n"
    "5. When certain, end with <answer>SHORT FINAL ANSWER ONLY</answer>.\n"
    "6. Special tokens must NOT appear in free text other than the function tags. Keep outputs minimal."
)

# ==========================================================
# Regex helpers
# ==========================================================
# CoA tool block and inner tools
TOOLS_BLOCK_RE = re.compile(r"<tools>(.*?)</tools>", re.S)
WEB_SEARCH_RE = re.compile(r"<web_search>(.*?)</web_search>", re.S)
CRAWL_PAGE_RE = re.compile(r"<crawl_page>(.*?)</crawl_page>", re.S)
# Six-function
WIKI_RE = re.compile(r"<wiki_search>(.*?)</wiki_search>", re.S)
# Shared
OBS_RE = re.compile(r"<observation>(.*?)</observation>", re.S)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

# ==========================================================
# Eval utilities (SQuAD-like normalization / EM / F1)
# ==========================================================
ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.I)


def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, golds: List[str]) -> int:
    p = normalize_text(pred)
    return int(any(p == normalize_text(g) for g in golds))


def f1_score(pred: str, golds: List[str]) -> float:
    def tok(t: str) -> List[str]:
        return normalize_text(t).split()

    pt = tok(pred)
    best = 0.0
    for g in golds:
        gt = tok(g)
        if not pt and not gt:
            return 1.0
        # multiset intersection
        common: Dict[str, int] = {}
        for t in pt:
            common[t] = common.get(t, 0) + 1
        inter = 0
        for t in gt:
            if common.get(t, 0) > 0:
                inter += 1
                common[t] -= 1
        if inter == 0:
            cur = 0.0
        else:
            prec = inter / len(pt) if pt else 0.0
            rec = inter / len(gt) if gt else 0.0
            cur = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best = max(best, cur)
    return best

# ==========================================================
# Message builders
# ==========================================================

def build_messages(question: str, proto: str) -> List[Dict[str, str]]:
    if proto == "coa":
        return [
            {"role": "system", "content": SYSTEM_COA},
            {"role": "user", "content": f"Question: {question}\nFollow the function protocol and finish with <answer>...</answer>."},
        ]
    elif proto == "sixfunc":
        return [
            {"role": "system", "content": SYSTEM_6F},
            {"role": "user", "content": f"Question: {question}\nFollow the function protocol and finish with <answer>...</answer>."},
        ]
    else:  # vanilla
        return [
            {"role": "system", "content": "You are a helpful assistant for open-domain QA. Answer concisely with a short factual phrase only."},
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ]

# ==========================================================
# Generation helpers
# ==========================================================

def apply_chat(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_gen_kwargs(args, tok) -> Dict[str, Any]:
    if args.do_sample or (args.temperature and args.temperature > 0):
        return dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            repetition_penalty=float(args.repetition_penalty),
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    else:
        return dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # greedy
            repetition_penalty=float(args.repetition_penalty),
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )


def generate_text(model, tok, prompt_text: str, gen_kwargs: Dict[str, Any]) -> str:
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen = tok.decode(out[0][input_len:], skip_special_tokens=True)
    return gen.strip()

# ==========================================================
# Simple web tools (best-effort, offline-friendly)
# ==========================================================

def simple_wiki_search(query: str, timeout: float = 3.0) -> str:
    """Wikipedia REST APIから短い要約を取得（512字に切詰）。
    インターネット不可の環境では空文字を返します。
    """
    try:
        import requests
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/" +
            requests.utils.quote(query.strip())
        )
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            j = r.json()
            txt = j.get("extract") or ""
            return (txt or "")[:512]
    except Exception:
        pass
    return ""


def simple_crawl_page(url: str, timeout: float = 4.0) -> str:
    """非常に簡易なページ取得（テキスト抽出して 700 字に切詰）。
    依存を増やさないために正規表現で HTML タグをざっくり剥がします。
    インターネット不可の環境では空文字。
    """
    try:
        import requests
        r = requests.get(url.strip(), timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            txt = r.text
            # 粗いタグ除去
            txt = re.sub(r"<script[\s\S]*?</script>", " ", txt, flags=re.I)
            txt = re.sub(r"<style[\s\S]*?</style>", " ", txt, flags=re.I)
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = re.sub(r"\s+", " ", txt)
            return txt.strip()[:700]
    except Exception:
        pass
    return ""

# ==========================================================
# Dialog runner (supports coa / sixfunc / vanilla)
# ==========================================================

@dataclass
class TurnTrace:
    turn: int
    prompt_preview: str
    output: str
    tool_queries: List[str]
    observation: str


def extract_tools_and_queries(proto: str, out: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """抽出したクエリのリストと (tool_type, payload) のタプル列を返す。
    tool_type は "web_search" / "crawl_page" / "wiki_search" のいずれか。
    """
    queries: List[str] = []
    calls: List[Tuple[str, str]] = []
    if proto == "coa":
        m_tb = TOOLS_BLOCK_RE.findall(out)
        for block in m_tb:
            for q in WEB_SEARCH_RE.findall(block):
                qq = q.strip()
                if qq:
                    queries.append(qq)
                    calls.append(("web_search", qq))
            for u in CRAWL_PAGE_RE.findall(block):
                uu = u.strip()
                if uu:
                    calls.append(("crawl_page", uu))
    elif proto == "sixfunc":
        for q in WIKI_RE.findall(out):
            qq = q.strip()
            if qq:
                queries.append(qq)
                calls.append(("wiki_search", qq))
    return queries, calls


def run_dialog(
    model,
    tok,
    question: str,
    proto: str,
    gen_kwargs: Dict[str, Any],
    allow_tools: bool,
    use_wiki: bool,
    max_turns: int,
) -> Tuple[str, List[TurnTrace], bool]:
    messages = build_messages(question, proto)
    traces: List[TurnTrace] = []

    for turn in range(1, max_turns + 1):
        prompt_text = apply_chat(tok, messages)
        out = generate_text(model, tok, prompt_text, gen_kwargs)

        tool_queries, calls = extract_tools_and_queries(proto, out)
        obs = ""

        # Handle tools
        if allow_tools and calls:
            # 1件目だけ処理（簡易）。複数ツールは順次ループでも良いがここでは単純化。
            ttype, payload = calls[0]
            if ttype in ("web_search", "wiki_search"):
                obs = simple_wiki_search(payload) if use_wiki else ""
            elif ttype == "crawl_page":
                obs = simple_crawl_page(payload)
            # Append assistant out + user observation
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"<observation>{obs}</observation>\nPlease continue."})
        else:
            # No tool call: just append
            messages.append({"role": "assistant", "content": out})

        traces.append(
            TurnTrace(
                turn=turn,
                prompt_preview=prompt_text[-4000:],
                output=out,
                tool_queries=tool_queries,
                observation=obs,
            )
        )

        # Answer?
        m_ans = ANS_RE.search(out)
        if m_ans:
            return m_ans.group(1).strip(), traces, True

    # Finalize
    messages.append({"role": "user", "content": "Finish with <answer>...</answer> now."})
    out = generate_text(model, tok, apply_chat(tok, messages), gen_kwargs)
    traces.append(TurnTrace(turn=max_turns + 1, prompt_preview="(finalize)", output=out, tool_queries=[], observation=""))
    m = ANS_RE.search(out)
    if m:
        return m.group(1).strip(), traces, True
    return out.strip(), traces, False

# ==========================================================
# LLM-as-Judge (local, lightweight)
# ==========================================================

JUDGE_PROMPT = (
    "You are a strict judge. Decide if the PREDICTION correctly answers the QUESTION.\n"
    "QUESTION: {q}\n"
    "PREDICTION: {p}\n"
    "(Optional) GOLD ANSWERS (may contain variants): {g}\n"
    "Reply with a single token: CORRECT or INCORRECT."
)


def llm_judge(judge_model, judge_tok, question: str, pred: str, golds: List[str]) -> int:
    msg = [
        {"role": "system", "content": "You are a careful evaluator."},
        {"role": "user", "content": JUDGE_PROMPT.format(q=question, p=pred, g=", ".join(map(str, golds))[:1200])},
    ]
    prompt = judge_tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    with torch.no_grad():
        inputs = judge_tok(prompt, return_tensors="pt").to(judge_model.device)
        out = judge_model.generate(**inputs, max_new_tokens=8, do_sample=False, eos_token_id=judge_tok.eos_token_id, pad_token_id=judge_tok.eos_token_id)
    text = judge_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return 1 if "CORRECT" in text and "INCORRECT" not in text else 0

# ==========================================================
# Main
# ==========================================================

def main():
    ap = argparse.ArgumentParser(description="Minimal MHQA evaluator with CoA tool loop and LLM-as-Judge option.")

    # Model/Data
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--tokenizer", default=None, help="Override tokenizer id (defaults to --model)")
    ap.add_argument("--dataset", default="nq_open", help="Dataset name (nq_open / natural_questions_open …)")
    ap.add_argument("--split", default="validation", help="Dataset split (validation/test)")
    ap.add_argument("--n", type=int, default=50, help="#samples for evaluation (random subset)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    # Protocol / dialog
    ap.add_argument("--proto", choices=["coa", "sixfunc", "vanilla"], default="coa", help="Response protocol")
    ap.add_argument("--max-turns", type=int, default=8, help="Max turns (>=1)")
    ap.add_argument("--no-tools", action="store_true", help="Disable tool handling (<tools>/<wiki_search>)")
    ap.add_argument("--use-wiki", action="store_true", help="Enable Wikipedia REST summary for web_search/wiki_search")

    # Generation
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)

    # Metrics
    ap.add_argument("--metric", choices=["emf1", "judge"], default="emf1", help="Evaluation metric")
    ap.add_argument("--judge-model", default=None, help="HF model id for LLM-as-Judge (required if --metric judge)")

    # Logging / output
    ap.add_argument("--out", default=None, help="Write per-sample JSONL of results")
    ap.add_argument("--save-trace", action="store_true", help="Write detailed trace JSONL alongside output")
    ap.add_argument("--trace-file", default=None, help="Explicit trace filename (defaults to <out>.trace.jsonl)")
    ap.add_argument("--trace-max-chars", type=int, default=4000, help="Max chars per trace field (0 = unlimited)")
    ap.add_argument("--out-runmeta", default=None, help="Write run meta JSON (defaults to <out>.runmeta.json)")
    ap.add_argument("--transformers-quiet", action="store_true")

    args = ap.parse_args()

    if args.transformers_quiet:
        hf_logging.set_verbosity_error()

    # Dataset
    print(f"Loading dataset: {args.dataset}, split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    idxs = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(idxs)
    idxs = idxs[: args.n]

    # Tokenizer / model
    print(f"Loading model: {args.model}")
    tok_repo = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tok_repo, trust_remote_code=True)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = "cuda"
    elif torch.backends.mps.is_available():
        dtype = torch.float32
        device = "mps"
    else:
        dtype = torch.float32
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    # Judge model (optional)
    judge_model = None
    judge_tok = None
    if args.metric == "judge":
        if not args.judge_model:
            raise SystemExit("--metric judge requires --judge-model")
        print(f"Loading judge model: {args.judge_model}")
        judge_tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)

    gen_kwargs = build_gen_kwargs(args, tok)

    allow_tools = not args.no_tools

    em_sum = 0.0
    f1_sum = 0.0
    acc_sum = 0.0

    results: List[Dict[str, Any]] = []

    # Trace file
    trace_fp = None
    trace_path = None
    if args.save_trace:
        if args.trace_file:
            trace_path = args.trace_file
        elif args.out:
            trace_path = args.out.rsplit(".", 1)[0] + ".trace.jsonl"
        else:
            trace_path = "run.trace.jsonl"
        trace_fp = open(trace_path, "w", encoding="utf-8")

    t0 = time.time()
    for k, i in enumerate(idxs, 1):
        q = ds[i]["question"]
        golds: List[str] = ds[i]["answers"] if "answers" in ds.features else ds[i]["answer"]

        pred, traces, ok = run_dialog(
            model,
            tok,
            question=q,
            proto=args.proto,
            gen_kwargs=gen_kwargs,
            allow_tools=allow_tools,
            use_wiki=args.use_wiki,
            max_turns=max(1, args.max_turns),
        )

        em = exact_match(pred, golds)
        f1 = f1_score(pred, golds)
        acc = em  # default
        if args.metric == "judge":
            acc = llm_judge(judge_model, judge_tok, q, pred, golds)

        em_sum += em
        f1_sum += f1
        acc_sum += acc

        print(
            f"[{k}/{len(idxs)}] Q: {q}\n  PRED: {pred}  (ok={ok})\n  GOLD: {golds[:3]}{' ...' if len(golds)>3 else ''}  EM={em} F1={f1:.3f} ACC={acc}"
        )

        rec = {"q": q, "pred": pred, "gold": golds, "em": em, "f1": f1, "acc": acc}
        results.append(rec)

        if trace_fp is not None:
            def cut(s: str) -> str:
                if args.trace_max_chars and args.trace_max_chars > 0 and len(s) > args.trace_max_chars:
                    return s[: args.trace_max_chars]
                return s

            trace_obj = {
                "index": i,
                "k": k,
                "question": q,
                "pred": pred,
                "ok": ok,
                "turns": [
                    {
                        "turn": t.turn,
                        "prompt_preview": cut(t.prompt_preview),
                        "output": cut(t.output),
                        "tool_queries": t.tool_queries,
                        "observation": cut(t.observation),
                    }
                    for t in traces
                ],
            }
            trace_fp.write(json.dumps(trace_obj, ensure_ascii=False) + "\n")
            trace_fp.flush()

    elapsed = time.time() - t0
    n_eval = len(idxs)
    em_avg = em_sum / n_eval
    f1_avg = f1_sum / n_eval
    acc_avg = acc_sum / n_eval

    print("\n==== SUMMARY ====")
    print(f"samples={n_eval}  EM={em_avg:.3f}  F1={f1_avg:.3f}  ACC={acc_avg:.3f}  time={elapsed:.1f}s")

    # Results JSONL
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved results to: {args.out}")

    # Runmeta
    out_runmeta = (
        args.out_runmeta
        if args.out_runmeta
        else (args.out.rsplit(".", 1)[0] + ".runmeta.json" if args.out else None)
    )
    if out_runmeta:
        meta = {
            "args": vars(args),
            "model": args.model,
            "tokenizer": getattr(tok, "name_or_path", None),
            "device": str(model.device),
            "dtype": str(model.dtype),
            "elapsed_sec": elapsed,
            "n_eval": n_eval,
            "em_avg": em_avg,
            "f1_avg": f1_avg,
            "acc_avg": acc_avg,
        }
        with open(out_runmeta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Saved run meta to: {out_runmeta}")

    if trace_fp is not None:
        trace_fp.close()
        print(f"Saved traces to: {trace_path}")


if __name__ == "__main__":
    main()
