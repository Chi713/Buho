import os
import json
import pathlib

# Path to dataset files
CONLLU_PATH = os.environ.get('BUHO_CONLLU_DATA_PATH')  # Replace with the actual dataset path
DATA_PATH = os.environ.get('BUHO_DATA_PATH')  # Replace with the actual dataset path
FILES = {
    "train": "es_ancora-ud-train.conllu",
    "dev": "es_ancora-ud-dev.conllu",
    "test": "es_ancora-ud-test.conllu"
}

def read_conllu(file_path):
    """
    Reads a CoNLL-U file and extracts sentences and corresponding POS tags.
    """
    sentences = []
    pos_tags = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        sentence_tokens = []
        sentence_tags = []
        
        for line in file:
            line = line.strip()
            
            if not line or line.startswith("#"):  # Skip comments and empty lines
                if sentence_tokens:  # End of a sentence
                    sentences.append(sentence_tokens)
                    pos_tags.append(sentence_tags)
                    sentence_tokens = []
                    sentence_tags = []
                continue
            
            fields = line.split("\t")
            if len(fields) < 4:  # Skip malformed lines
                continue
            
            token = fields[1]  # The word token
            pos_tag = fields[3]  # The POS tag (UPOS)
            
            sentence_tokens.append(token)
            sentence_tags.append(pos_tag)
        
        # Append last sentence if file does not end with a newline
        if sentence_tokens:
            sentences.append(sentence_tokens)
            pos_tags.append(sentence_tags)
    
    return sentences, pos_tags

def save_to_file(data, output_file):
    """
    Saves the processed data to a JSON file.
    """
    
    # output_file_path = os.path.join(DATA_PATH, "data_cache")
    output_file_path = DATA_PATH
    pathlib.Path(output_file_path).mkdir(parents=True, exist_ok=True) 
    print(output_file_path)
    output_file = os.path.join(output_file_path, output_file)
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
        
        sentences, pos_tags = read_conllu(file_path)
        data = {
            "sentences": sentences,
            "pos_tags": pos_tags
        }
        
        output_file = f'{split_name}_data.json'
        save_to_file(data, output_file)

def main():
    process_and_save(CONLLU_PATH, FILES)

if __name__ == "__main__":
    main()
