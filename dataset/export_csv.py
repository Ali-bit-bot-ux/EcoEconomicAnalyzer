import json
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def json_to_csv(json_name, csv_name, list_keys_to_join=None):
    if list_keys_to_join is None:
        list_keys_to_join = []
    
    json_path = os.path.join(BASE_DIR, json_name)
    csv_path = os.path.join(BASE_DIR, csv_name)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not data:
        return
        
    flat_data = []
    for item in data:
        row = {}
        for k, v in item.items():
            if isinstance(v, list):
                if all(isinstance(x, str) for x in v):
                    row[k] = "; ".join(v)
                else:
                    row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
        flat_data.append(row)
        
    fieldnames = list(flat_data[0].keys())
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_data)
    print(f"Generated {csv_name} from {json_name} ({len(flat_data)} rows)")

if __name__ == '__main__':
    json_to_csv("soils.json", "soils.csv")
    json_to_csv("drought_practices.json", "drought_practices.csv")
    json_to_csv("crops.json", "crops.csv")
    json_to_csv("soil_crop_mitigation_matrix.json", "soil_crop_mitigation_matrix.csv")
    json_to_csv("references.json", "references.csv")
