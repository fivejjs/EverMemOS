import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from time import time

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

# Ensure project root is on sys.path so `evaluation` can be imported when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.locomo_evaluation.config import ExperimentConfig
from evaluation.locomo_evaluation.prompts import ANSWER_PROMPT_NEMORI, ANSWER_PROMPT_NEMORI_COT

async def locomo_response(llm_client, llm_config, context: str, question: str, experiment_config: ExperimentConfig) -> str:
    if experiment_config.mode == "cot":
        prompt = ANSWER_PROMPT_NEMORI_COT.format(
            context=context,
            question=question,
        )
    else:
        prompt = ANSWER_PROMPT_NEMORI.format(
            context=context,
            question=question,
        )
    for i in range(experiment_config.max_retries):
        try:
            response = await llm_client.chat.completions.create(
                model=llm_config["model"],
                messages=[
                    {"role": "system", "content": prompt},
                ],
                # temperature=llm_config["temperature"],
                temperature=0,
                # max_tokens=llm_config["max_tokens"],
                max_tokens=4096,
            )
            result = response.choices[0].message.content or ""
            if experiment_config.mode == "cot":
                result = result.split("FINAL ANSWER:")[1].strip()
            if result == "":
                continue
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    return result


async def process_qa(qa, search_result, oai_client, llm_config, experiment_config):
    start = time()
    query = qa.get("question")
    gold_answer = qa.get("answer")
    qa_category = qa.get("category")

    answer = await locomo_response(oai_client, llm_config, search_result.get("context"), query, experiment_config)

    response_duration_ms = (time() - start) * 1000

    print(f"Processed question: {query}")
    print(f"Answer: {answer}")
    return {
        "question": query,
        "answer": answer,
        "category": qa_category,
        "golden_answer": gold_answer,
        "search_context": search_result.get("context", ""),
        "response_duration_ms": response_duration_ms,
        "search_duration_ms": search_result.get("duration_ms", 0),
    }


async def main(search_path, save_path):
    llm_config = ExperimentConfig.llm_config["openai"]
    experiment_config = ExperimentConfig()
    oai_client = AsyncOpenAI(
        api_key=llm_config["api_key"], base_url=llm_config["base_url"]
    )
    locomo_df = pd.read_json(experiment_config.datase_path)
    with open(search_path) as file:
        locomo_search_results = json.load(file)

    num_users = len(locomo_df)

    all_responses = {}
    for group_idx in range(num_users):
        qa_set = locomo_df["qa"].iloc[group_idx]
        qa_set_filtered = [qa for qa in qa_set if qa.get("category") != 5]

        group_id = f"locomo_exp_user_{group_idx}"
        search_results = locomo_search_results.get(group_id)

        matched_pairs = []
        for qa in qa_set_filtered:
            question = qa.get("question")
            matching_result = next(
                (result for result in search_results if result.get("query") == question), None
            )
            if matching_result:
                matched_pairs.append((qa, matching_result))
            else:
                print(f"Warning: No matching search result found for question: {question}")

        tasks = [
            process_qa(qa, search_result, oai_client, llm_config, experiment_config)
            for qa, search_result in tqdm(
                matched_pairs,
                desc=f"Processing {group_id}",
                total=len(matched_pairs),
            )
        ]

        responses = await asyncio.gather(*tasks)
        all_responses[group_id] = responses

    os.makedirs("data", exist_ok=True)

    print(all_responses)

    with open(save_path, "w") as f:
        json.dump(all_responses, f, indent=2)
        print("Save response results")


if __name__ == "__main__":
    config = ExperimentConfig()
    search_result_path = str(Path(__file__).parent / "results" / config.experiment_name / "search_results.json")
    save_path = Path(__file__).parent / "results" / config.experiment_name / "responses.json"
    # search_result_path = f"/Users/admin/Documents/Projects/b001-memsys/evaluation/locomo_evaluation/results/locomo_evaluation_0/nemori_locomo_search_results.json"
    
    asyncio.run(main(search_result_path, save_path))
