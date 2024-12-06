import os
import json
import pathlib
import pyconll

# Path to dataset files
CONLLU_PATH = os.environ.get('BUHO_CONLLU_DATA_PATH')  # Replace with the actual dataset path
DATA_PATH = os.environ.get('BUHO_DATA_PATH')  # Replace with the actual dataset path
FILES = {
    "train": "es_ancora-ud-train.conllu",
    "dev": "es_ancora-ud-dev.conllu",
    "test": "es_ancora-ud-test.conllu"
}

# POS Mapping
POS_TO_ID = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5, 'INTJ': 6, 'NOUN': 7, 'NUM': 8, 
             'PART': 9, 'PRON': 10, 'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'SYM': 14, 'VERB': 15, 'X': 16, '_': 17}
ID_TO_POS = {v: k for k, v in POS_TO_ID.items()}

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

def process_and_save(dataset_path, files):
    """
    Processes each dataset split and saves them to separate files.
    """
    for split_name, file_name in files.items():
        file_path = os.path.join(dataset_path, file_name)
        print(f"Processing {split_name} data from {file_path}...")

        conll = pyconll.load_from_file(file_path)
        sentences = [[token.form for token in sentence] for sentence in conll]
        lemmas = [[(token.lemma if token.lemma is not None else "_") for token in sentence] for sentence in conll]
        pos_tags = [[(token.upos if token.upos is not None else "_") for token in sentence] for sentence in conll]

        data = {
            "sentences": sentences,
            "lemmas": lemmas,
            "pos_tags": [[POS_TO_ID[tag] for tag in tags] for tags in pos_tags]
        }
        
        output_file = f'{split_name}_data.json'
        print(output_file)
        
        save_to_file(data, output_file)

def main():
    process_and_save(CONLLU_PATH, FILES)

if __name__ == "__main__":
    print(CONLLU_PATH)
    main()