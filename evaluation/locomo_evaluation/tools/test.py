import json

with open("/Users/admin/Documents/Projects/b001-memsys/evaluation/locomo_evaluation/results/locomo_evaluation_4/memcell_list_conv_0.json", "r") as f:
    data = json.load(f)

for item in data:
    print(len(item['original_data']))
