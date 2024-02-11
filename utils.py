import os
import json

def safe_open_w(path: str) -> object:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def create_path(path : str) -> None:
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f'No such dir: {path}'

def dataset_stats(dataset_path : str) -> None:
    dataset = json.load(open(dataset_path, encoding="utf8"))

    samples = len(dataset)
    labels = {"Entailment": 0, "Contradiction": 0}
    types = {"Single" : 0, "Comparison" : 0}
    sections = {"Adverse Events" : 0, "Eligibility" : 0, "Intervention" : 0, "Results" : 0}
    interventions_done = {"Paraphrase" : 0, "Contradiction" : 0, "Text_appended" : 0, "Numerical_contradiction" : 0, "Numerical_paraphrase" :0}
    type_of_intervention = {"Preserving" : 0, "Altering" : 0}

    for query in dataset:
        if "Type" in dataset[query]:
            types[dataset[query]["Type"]] += 1
        if "Section_id" in dataset[query]:
            sections[dataset[query]["Section_id"]] += 1
        if "Intervention" in dataset[query]:
            interventions_done[dataset[query]["Intervention"]] += 1
        if "Causal_type" in dataset[query]:
            type_of_intervention[dataset[query]["Causal_type"][0]] += 1
        if "Label" in dataset[query]:
            labels[dataset[query]["Label"]] += 1

    output_prints = [f"# of samples: {samples}"]
    for stats in [labels, types, sections, interventions_done, type_of_intervention]:
        output_prints.append("")
        for key in stats:
            output_prints[-1]+=(f"{key}: {stats[key]} ({stats[key]/samples*100:.2f}%) | ")
        output_prints[-1] = output_prints[-1][:-3]
    for line in output_prints:
        print(line)

def count_corpus_stats(corpus_path : str) -> None:
    corpus = json.load(open(corpus_path, encoding="utf8"))
    corpus_words = sum([len(" ".join(corpus[doc][sec]).split(" ")) for doc in corpus for sec in corpus[doc]])
    print(f"Corpus size: {len(corpus)}")
    print(f"Corpus words: {corpus_words} | {corpus_words//len(corpus)} words per document")

#dataset_stats('qrels/qrels2024_train-synthetic.json')