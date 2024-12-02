import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import time

# define model path
MODELS_PATH = os.environ.get('BUHO_MODEL_PATH')
MODEL_PATH = os.path.join(MODELS_PATH, "pos_tagging_model")

# Define ID_TO_POS mapping
ID_TO_POS = {
    0: 'ADJ', 1: 'ADP', 2: 'ADV', 3: 'AUX', 4: 'CCONJ',
    5: 'DET', 6: 'INTJ', 7: 'NOUN', 8: 'NUM', 9: 'PART',
    10: 'PRON', 11: 'PROPN', 12: 'PUNCT', 13: 'SCONJ',
    14: 'SYM', 15: 'VERB', 16: 'X', 17: '_'
}
# POS Mapping
POS_TO_ID = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5, 'INTJ': 6, 'NOUN': 7, 'NUM': 8, 
             'PART': 9, 'PRON': 10, 'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'SYM': 14, 'VERB': 15, 'X': 16, '_': 17}

# Load model and tokenizer
start_time = time.time()
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
#model = AutoModelForTokenClassification.from_pretrained(
#    "dccuchile/bert-base-spanish-wwm-cased", num_labels=len(POS_TO_ID)
#)
model.eval()
load_time = time.time()

# Test sentence
# Sentence split into words
sentence =         [
            "Partidario",
            "de",
            "la",
            "\"",
            "perestroika",
            "\"",
            "de",
            "Mijail",
            "Gorbachov",
            "en",
            "la",
            "Unión",
            "Soviética",
            ",",
            "en",
            "1989",
            "entró",
            "en",
            "conflicto",
            "con",
            "Yívkov",
            ",",
            "líder",
            "durante",
            "35",
            "años",
            "del",
            "de",
            "el",
            "Partido",
            "Comunista",
            "y",
            "del",
            "de",
            "el",
            "Estado",
            "búlgaro",
            ",",
            "y",
            "le",
            "acusó",
            "en",
            "una",
            "carta",
            "abierta",
            "de",
            "utilizar",
            "métodos",
            "poco",
            "democráticos",
            "de",
            "gobierno",
            "."
        ]
text = 'Partidario de la "perestroika" de Mijail Gorbachov en la Unión Soviética, en 1989 entró en conflicto con Yívkov, líder durante 35 años del Partido Comunista y del Estado búlgaro, y le acusó en una carta abierta de utilizar métodos poco democráticos de gobierno.'


# Tokenize input with `is_split_into_words`
inputs = tokenizer(text, is_split_into_words=False, return_tensors="pt", truncation=True)

# Print tokenized output for verification
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("Tokens:", tokens)
print("perestroika" in tokenizer.get_vocab())
print("Mijail" in tokenizer.get_vocab())
print("Gorbachov" in tokenizer.get_vocab())

tokenize_time = time.time()

# Run inference
outputs = model(**inputs)
tagging_time = time.time()

# Decode predictions
predictions = torch.argmax(outputs.logits, dim=-1)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Convert input IDs to tokens
predicted_tags = [predictions[0][i].item() for i in range(len(tokens))]

# Display the results
for token, tag in zip(tokens, predicted_tags):
    # Skip special tokens
    if token in ["[CLS]", "[SEP]"]:
        continue
    print(f"{token}: {ID_TO_POS.get(tag, 'UNKNOWN')}")
print(f"tokenizing time: {load_time - start_time}")
print(f"tokenizing time: {tokenize_time - load_time}")
print(f"tagging time: {tagging_time - tokenize_time}")
print(f"total time: {tagging_time - start_time}")
