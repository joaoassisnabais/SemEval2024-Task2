import json
from transformers import AutoTokenizer
from collections import Counter
import matplotlib.pylab as plt

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

def create_dataset():
    # Load the JSON files
    mistake_file = load_json(mistake_file_path)
    cot = load_json(cot_file_path)

    if mistake_file and cot:
        mistakes = mistake_file["mistakes"]
        for qid in mistakes:
            if qid['q_id'] in cot:
                cot.pop(qid['q_id'])

    #for qid in cot:
    #    print(qid)

    with open('qrels/2025-01-28_20-23_train-manual-expand_and_dev-cot.json', 'w') as file:
        file.write(json.dumps(cot, indent=4))

def avg_qrel_length():
    qrels = load_json('qrels/2025-01-28_20-23_train-manual-expand_and_dev-cot.json')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', padding_side="left")
    
    total_sum = 0
    len_qrels = []
    # Calculate the average length of the QRELs
    for cot in qrels.values():
        tokens = tokenizer.encode(cot)
        total_sum += len(tokens)
        len_qrels.append(len(tokens))
    
    c_qrels = Counter(len_qrels)
    list_x = list(c_qrels.keys())
    list_y = list(c_qrels.values())
    
    plt.bar(list_x, list_y)
    plt.xlabel('Length of QRELs')
    plt.ylabel('Count')
    plt.title('Distribution of QREL Lengths')
    plt.show()
    
    avg_length = total_sum / len(qrels)
    print(f"Average QREL length: {avg_length}")
    

if __name__ == "__main__":
    avg_qrel_length()
    
