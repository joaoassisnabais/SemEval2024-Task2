import json
import argparse
from utils import safe_open_w
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to input dir/file', default='qrels/')
    parser.add_argument('--output', type=str, help='path to output dir/file', default='queries/')
    parser.add_argument('--corpus', type=str, help='path to CT Corpus', default='CT_json/SemEval_CT-corpus.json')
    args = parser.parse_args() 

    qrels = json.load(open(args.input))
    corpus = json.load(open(args.corpus, encoding="utf8"))

    output_dict = {}

    #TO:DO 
    for query in tqdm(qrels):  
        output_dict[query] = {}

        if "TO DO" in qrels[query]["Statement"]:
            neg_example = query.split("_")[0]+"_neg"
            if neg_example in output_dict:
                del output_dict[neg_example]
            del output_dict[query]
            continue

        for data_field in ['Type', 'Section_id', 'Primary_id', 'Secondary_id', 'Section_id', 'Statement']:
            if data_field in qrels[query]:
               output_dict[query][data_field] = qrels[query][data_field]
                    
        output_dict[query]['Primary_id_txt_list'] = corpus[output_dict[query]['Primary_id']][output_dict[query]['Section_id']]
        output_dict[query]['Primary_id_txt'] = "\n".join(line for line in output_dict[query]['Primary_id_txt_list'])
        if 'Secondary_id' in output_dict[query]:
            output_dict[query]['Secondary_id_txt_list'] = corpus[output_dict[query]['Secondary_id']][output_dict[query]['Section_id']]
            output_dict[query]['Secondary_id_txt'] = "\n".join(line for line in output_dict[query]['Secondary_id_txt_list'])

    with safe_open_w(args.output) as f:
        json.dump(output_dict, f, indent=4)