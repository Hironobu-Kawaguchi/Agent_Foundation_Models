# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import random
import re
import string


def _log_print(data_source, prompt_str, solution_str, ground_truth, extracted_ans, reward, extra_info=None):
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using EM scoring method")
        print(f"Data source: {data_source}")
        print(f"Prompt string: {prompt_str}")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {extracted_ans}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Reward: {reward}")
        if extra_info is not None:
            print(f"Extra info: {extra_info}")
        print(f"--------------------------------")


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(predictions, golden_answers):
    # 确保黄金答案是列表格式
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    # 标准化所有黄金答案
    normalized_golden = [normalize_answer(ans) for ans in golden_answers]
    
    # 先去空
    predictions = [prediction for prediction in predictions if prediction]

    # 处理预测为空的情况（空预测视为不符合要求）
    if not predictions:
        return 0
    
    # 检查每个预测是否都符合要求
    for prediction in predictions:
        normalized_pred = normalize_answer(prediction)
        # 只要有一个预测不在黄金答案中，就返回0
        if normalized_pred not in normalized_golden:
            return 0
    # 所有预测都符合要求，返回1
    return 1


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]
    answer_pattern = r'<answer>(.*?)</answer>'
    # 查找所有匹配的标签内容，使用DOTALL模式确保.匹配换行符
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if not matches:
        return None
    
    # 取最后一个匹配的内容（不使用strip()，避免移除有意义的空白）
    last_content = matches[-1].group(1)
    # 按|分割成列表
    answers = last_content.split('|')
    
    all_answers = answers if answers else None
    
    return all_answers


def compute_score_em(data_source, prompt_str, solution_str, ground_truth, extra_info=None, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    all_answers = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    rw = 0
    if all_answers is None:
        rw = 0
    else:
        if em_check(all_answers, ground_truth['target']):
            rw = score
        else:
            rw = format_score
    do_print = False
    if do_print:
        _log_print(data_source, prompt_str, solution_str, ground_truth, all_answers, rw, extra_info)

    return {
        "score": rw,
        "em": rw
    }

def compute_score_em_batch(
    data_sources: list[str],
    prompt_strs: list[str],
    solution_strs: list[str], 
    ground_truths: list[dict],
    extra_infos: list[dict],
    **kwargs
) -> list[float]:

    scores = []
    for (ds, p, s, gt) in zip(data_sources, prompt_strs, solution_strs, ground_truths):
        score = compute_score_em(ds, p, s, gt).get("score")
        scores.append(score)
    
    return scores
def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score