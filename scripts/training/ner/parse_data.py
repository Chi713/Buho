import os
import json
import pathlib
# import pyconll

# Path to dataset files
#CONLLU_PATH = os.environ.get('BUHO_POS_CONLLU_DATA_PATH')  # Replace with the actual dataset path
DATA_PATH = os.environ.get('BUHO_DATA_PATH')  # Replace with the actual dataset path
# FILES = {
#     "train": "es_ancora-ud-train.conllu",
#     "dev": "es_ancora-ud-dev.conllu",
#     "test": "es_ancora-ud-test.conllu"
# }

from datasets import load_dataset
datasets = {
    "train": load_dataset('eriktks/conll2003', split='train', trust_remote_code=True),
    "dev": load_dataset('eriktks/conll2003', split='validation', trust_remote_code=True),
    "test": load_dataset('eriktks/conll2003', split='test', trust_remote_code=True)
}

# NER Mapping
NER_TO_ID = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ID_TO_NER = {v: k for k, v in NER_TO_ID.items()}

def save_to_file(data, output_file):
    """
    Saves the processed data to a JSON file.
    """
    # output_file_path = os.path.join(DATA_PATH, "data_cache")
    output_file_path = DATA_PATH
    pathlib.Path(output_file_path).mkdir(parents=True, exist_ok=True) 
    print(output_file_path)
    output_file = os.path.join(output_file_path, output_file)
    print(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Saved {output_file} with {len(data['sentences'])} sentences.")

def process_and_save(datasets):
    """
    Processes each dataset split and saves them to separate files.
    """
    for dataset_name, dataset in datasets.items():
        # sentences = [[token.form for token in sentence] for sentence in conll]
        # lemmas = [[(token.lemma if token.lemma is not None else "_") for token in sentence] for sentence in conll]
        # pos_tags = [[(token.upos if token.upos is not None else "_") for token in sentence] for sentence in conll]

        # sentences = 

        data = {
            "sentences": dataset["tokens"],
            "ner_tags": dataset["ner_tags"],
        }
        
        output_file = f'{dataset_name}_ner_data.json'
        # print(data)
        
        save_to_file(data, output_file)

def main():
    process_and_save(datasets)

if __name__ == "__main__":
    # print(CONLLU_PATH)
    main()
