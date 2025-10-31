import json
from pathlib import Path
import os
from collections import defaultdict

def convert_nemori_to_memsys():
    nemori_data_path = "/Users/admin/Documents/Projects/b001-memsys_/evaluation/locomo_evaluation/results/locomo_evaluation_nemori/final_processed_episodes.json"
    output_dir = Path("/Users/admin/Documents/Projects/b001-memsys_/evaluation/locomo_evaluation/results/locomo_evaluation_nemori/")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(nemori_data_path, "r") as f:
        nemori_data = json.load(f)

    conversations = {}
    for key, value in nemori_data.items():
        parts = key.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            conv_id = int(parts[-1])
            if conv_id not in conversations:
                conversations[conv_id] = value

    for conv_id, data in conversations.items():
        after_processed_data = []
        for item in data:
            after_processed_data.append({
                "event_id": item["episode_id"],
                "original_data": item["raw_data_items"],
                "timestamp": item["timestamp"],
                "subject": item["title"],
                "summary": item["summary"],
                "episode": item["content"],
            })
        output_path = output_dir / f"memcell_list_conv_{conv_id}.json"
        with open(output_path, "w") as f:
            json.dump(after_processed_data, f, indent=4)
        print(f"Saved conversation {conv_id} to {output_path}")

if __name__ == "__main__":
    convert_nemori_to_memsys() 