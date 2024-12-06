import os
import json
import pathlib
import pyconll

# Path to dataset files
CONLLU_PATH = os.environ.get('BUHO_POS_CONLLU_DATA_PATH')  # Replace with the actual dataset path
DATA_PATH = os.environ.get('BUHO_DATA_PATH')  # Replace with the actual dataset path
FILES = {
    "train": "es_ancora-ud-train.conllu",
    "dev": "es_ancora-ud-dev.conllu",
    "test": "es_ancora-ud-test.conllu"
}

# POS Mapping
POS_TO_ID = {'-PAD-': 0,'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 
             'PART': 10, 'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, '_': 18 }
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

    print(f"Saved {output_file} with {len(data)} sentences.")

def process_and_save(dataset_path, files):
    """
    Processes each dataset split and saves them to separate files.
    """
    for split_name, file_name in files.items():
        file_name = "es_ancora-ud-train.conllu"
        
        file_path = os.path.join(dataset_path, file_name)
        print(f"Processing lemma data from {file_path}...")

        conll = pyconll.load_from_file(file_path)
        sentences = [
            [
                {token.form + "+" + (token.upos if token.upos is not None else "_") :
                (token.lemma if token.lemma is not None else "_")} 
                for token in sentence]
            for sentence in conll]

        flattened_sentences = [token_dict for sentence in sentences for token_dict in sentence]

        merged_sentences = {}
        for token_dict in flattened_sentences:
            merged_sentences.update(token_dict)  # Combine dictionaries, overwriting duplicates


        output_file = 'lemma_data.json'
        print(output_file)
        
        save_to_file(merged_sentences, output_file)

def main():
    process_and_save(CONLLU_PATH, FILES)

if __name__ == "__main__":
    print(CONLLU_PATH)
    main()
