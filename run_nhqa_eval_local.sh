!/bin/bash

# 論文準拠・Qwen2.5-3B-Instruct・greedy、vanilla、一発回答
uv run python run_mhqa_eval_local.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --n 50 \
  --proto vanilla \
  --max-turns 1 --no-tools \
  --out out_nq_qwen_vanilla.jsonl \
  --transformers-quiet

# 実行例（Qwen2.5-3B-Instruct，6-function、ツール無し）
uv run python run_mhqa_eval_local.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --n 50 \
  --proto sixfunc \
  --max-turns 8 --no-tools \
  --out out_nq_qwen_sixfunc.jsonl \
  --save-trace --trace-max-chars 0 \
  --transformers-quiet

# 論文準拠 AFM-SFT 6-function、--use-wiki
uv run python run_mhqa_eval_local.py \
  --model PersonalAILab/AFM-MHQA-Agent-3B-sft \
  --n 50 \
  --proto sixfunc \
  --max-turns 8 --use-wiki \
  --out out_nq_sft.jsonl \
  --save-trace --trace-max-chars 0

# 論文準拠 AFM-RL 6-function、--use-wiki
uv run python run_mhqa_eval_local.py \
  --model PersonalAILab/AFM-MHQA-Agent-3B-rl \
  --n 50 \
  --proto sixfunc \
  --max-turns 8 --use-wiki \
  --out out_nq_rl.jsonl \
  --save-trace --trace-max-chars 0

# 論文準拠 AFM-RL CoA、--use-wiki
uv run python run_mhqa_eval_local.py \
    --model PersonalAILab/AFM-MHQA-Agent-3B-rl \
    --dataset nq_open --split validation --n 50 --seed 42 \
    --proto coa --use-wiki \
    --metric emf1 \
    --do-sample --temperature 0.8 --top-p 0.9 --top-k 20 \
    --out out_nq_rl.jsonl --save-trace --trace-max-chars 0

# 旧 six-function 互換 + EM/F1
uv run python run_mhqa_eval_local.py \
    --model PersonalAILab/AFM-MHQA-Agent-3B-rl \
    --n 50 --proto sixfunc --use-wiki \
    --metric emf1 --out out_nq_emf1.jsonl
