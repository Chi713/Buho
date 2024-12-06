
import os
import json
# from scripts.utils.fragment_assembler import assemble

# Paths to tokenized data
DATA_PATH = os.environ.get('BUHO_DATA_PATH')
MODELS_PATH = os.environ.get('BUHO_MODELS_PATH')
MODEL_NAME = os.environ.get('BUHO_MODEL_NAME')

MODEL_PATH =  os.path.join(MODELS_PATH, MODEL_NAME)

# POS Mapping
# POS_TO_ID = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5, 'INTJ': 6, 'NOUN': 7, 'NUM': 8, 
#              'PART': 9, 'PRON': 10, 'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'SYM': 14, 'VERB': 15, 'X': 16, '_': 17}
# ID_TO_POS = {v: k for k, v in POS_TO_ID.items()}

# Load your data
def load_data(file_path):
    """Load JSON data from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

TRAIN_FILE = os.path.join(DATA_PATH, "lemma_data.json")
lemma_dict = load_data(TRAIN_FILE)

# def lemmatize(tokens: list[str], pos_tags: list[str]):
#     for (token, tag) in zip(tokens, pos_tags):
#         key = token + "+" + tag
#         if key in lemma_dict:
#             return lemma_dict[key]
#         else:
#             return "NO_LEMMA"
            
def lemmatize(token: str, tag: str):
    key = token + "+" + tag
    if token.endswith("s") and tag in ["NOUN", "ADJ"]:
        token = token[:-1]
    elif token.endswith("es") and tag in ["NOUN", "ADJ"]:
        token = token[:-2]
    if key in lemma_dict:
        return lemma_dict[key]
    else:
        return "NO_LEMMA"

