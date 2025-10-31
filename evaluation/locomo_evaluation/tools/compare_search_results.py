import json
from pathlib import Path

def compare_contexts():
    memsys_file = Path("/Users/admin/Documents/Projects/b001-memsys_/evaluation/locomo_evaluation/results/locomo_evaluation_nemori/nemori_locomo_search_results_nemori.json")
    nemori_file = Path("/Users/admin/Documents/Projects/b001-memsys_/evaluation/locomo_evaluation/results/locomo_evaluation_nemori/nemori_locomo_search_results.json")

    with open(memsys_file, 'r') as f:
        memsys_data = json.load(f)

    with open(nemori_file, 'r') as f:
        nemori_data = json.load(f)

    # Assuming the top-level keys are user IDs that match between files
    all_user_ids = set(memsys_data.keys()) | set(nemori_data.keys())

    for user_id in all_user_ids:
        print(f"--- Comparing results for user: {user_id} ---")
        
        memsys_results = memsys_data.get(user_id, [])
        nemori_results = nemori_data.get(user_id, [])

        memsys_map = {item['query']: item['context'] for item in memsys_results}
        nemori_map = {item['query']: item['context'] for item in nemori_results}

        all_queries = set(memsys_map.keys()) | set(nemori_map.keys())

        differences_found = False
        for query in sorted(list(all_queries)):
            memsys_context = memsys_map.get(query)
            nemori_context = nemori_map.get(query)

            if memsys_context is None:
                print(f"Query found only in nemori_locomo_search_results.json: '{query}'")
                differences_found = True
                continue
            
            if nemori_context is None:
                print(f"Query found only in nemori_locomo_search_results_memsys.json: '{query}'")
                differences_found = True
                continue

            if memsys_context.strip() != nemori_context.strip():
                print(f"Context differs for query: '{query}'")
                # Optional: print the differing contexts for detailed analysis
                # print("Memsys context:", memsys_context)
                # print("Nemori context:", nemori_context)
                differences_found = True

        if not differences_found:
            print("All contexts for this user are identical.")
        
        print("\n")


if __name__ == "__main__":
    compare_contexts()
