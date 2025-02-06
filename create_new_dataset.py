import json

# Define the path to the JSON file
mistake_file_path = 'outputs/mistakes/2025-01-28_20-23_Meta-Llama-3.1-70B-Instruct-quantized.w4a16_train-manual-expand_and_dev-set.json'
cot_file_path = 'outputs/2025-01-28_20-23_train-manual-expand_and_dev-set_full_output.json'

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

# Load the JSON files
mistake_file = load_json(mistake_file_path)
cot = load_json(cot_file_path)

if mistake_file and cot:
    mistakes = mistake_file["mistakes"]
    for qid in mistakes:
        if qid['q_id'] in cot:
            cot.pop(qid['q_id'])


with open('qrels/2025-01-28_20-23_train-manual-expand_and_dev-cot.json', 'w') as file:
    json.dump(cot, file, indent=4)

    
