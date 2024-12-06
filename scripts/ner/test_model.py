import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import time
import json

def assemble(tokens: list[str], predictions: list[str], answers: list[str]):
    tokens.reverse()
    predictions.reverse()
    answers.reverse()

    assembled_tokens = []
    new_predictions = []
    new_answers = []
    temp_token = ""    
    for token, prediction, answer in zip(tokens, predictions, answers):
        if token.startswith('##'):
            temp_token = token[2:] + temp_token
        else:
            token = token + temp_token
            assembled_tokens.append(token)
            new_predictions.append(prediction)
            new_answers.append(answer)
            print(f"token: {token}")
            temp_token = ""
    assembled_tokens.reverse()
    new_predictions.reverse()
    new_answers.reverse()

    print(assembled_tokens, new_predictions, new_answers)
    return assembled_tokens, new_predictions, new_answers

# import sys
# sys.path.append("..")
# from utils.fragment_assembler import assemble
# define model path
# from utils.fragment_assembler import assemble
DATA_PATH = os.environ.get('BUHO_DATA_PATH')  # Replace with the actual dataset path
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH') #+ '_v2'
NER_MODEL_PATH = os.path.join(MODEL_PATH, "ner")
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")

TRAIN_PATH = os.path.join(DATA_PATH, "train_inputs_ner.pt")
TEST_FILE = os.path.join(DATA_PATH, "test_ner_data.json")
def load_data(file_path):
    """Loads the JSON data file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data(TEST_FILE)

# def insert_return(ins_list=[]): 
#     ins_list.insert(0,-100) # add initial [CLS] bert token
#     ins_list.append(-100) # add final [SEP] bert token
#     return ins_list

sentence = [ ner_tags_list for ner_tags_list in data['sentences']][0]
answers = [ ner_tags_list for ner_tags_list in data['ner_tags']][0]
# Define ID_TO_NER mapping
NER_TO_ID = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ID_TO_NER = {v: k for k, v in NER_TO_ID.items()}


# Load model and tokenizer
start_time = time.time()

model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)
# tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
# model = AutoModelForTokenClassification.from_pretrained(
#    "dccuchile/bert-base-spanish-wwm-cased", num_labels=len(NER_TO_ID)
#)
model.eval()
load_time = time.time()

# Test sentence
# Sentence split into words
# sentence =         [
#             "Partidario",
#             "de",
#             "la",
#             "\"",
#             "perestroika",
#             "\"",
#             "de",
#             "Mijail",
#             "Gorbachov",
#             "en",
#             "la",
#             "Unión",
#             "Soviética",
#             ",",
#             "en",
#             "1989",
#             "entró",
#             "en",
#             "conflicto",
#             "con",
#             "Yívkov",
#             ",",
#             "líder",
#             "durante",
#             "35",
#             "años",
#             "del",
#             "de",
#             "el",
#             "Partido",
#             "Comunista",
#             "y",
#             "del",
#             "de",
#             "el",
#             "Estado",
#             "búlgaro",
#             ",",
#             "y",
#             "le",
#             "acusó",
#             "en",
#             "una",
#             "carta",
#             "abierta",
#             "de",
#             "utilizar",
#             "métodos",
#             "poco",
#             "democráticos",
#             "de",
#             "gobierno",
#             "."
#         ]
# text = 'Partidario de la "perestroika" de Mijail Gorbachov en la Unión Soviética, en 1989 entró en conflicto con Yívkov, líder durante 35 años del Partido Comunista y del Estado búlgaro, y le acusó en una carta abierta de utilizar métodos poco democráticos de gobierno.'

# Tokenize input with `is_split_into_words`
print(sentence)
print(len(answers))
inputs = tokenizer(sentence, is_split_into_words=True, return_tensors="pt", truncation=True)
print(f"answers tokenized")

labels = []
# for i, sentence in enumerate(sentences):
word_ids = inputs.word_ids()  # Map subword tokens to original words
label_ids = []
previous_word_idx = None

for word_idx in word_ids:
    if word_idx is None:
        label_ids.append(0)
    elif word_idx == previous_word_idx:
        # Set -100 for padding and subword tokens (ignored during loss computation)
        label_ids.append(0)
    else:
        label_ids.append(answers[word_idx])  # Align NER tag to the original word
    previous_word_idx = word_idx

labels.append(label_ids)
# answers = [ insert_return(ner_tags_list) for ner_tags_list in labels][0]
print(answers)


# Print tokenized output for verification
# tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
# print("Tokens:", tokens)
# print("Soviética" in tokenizer.get_vocab())
# print("Mijail" in tokenizer.get_vocab())
# print("Gorbachov" in tokenizer.get_vocab())

tokenize_time = time.time()

# Run inference
outputs = model(**inputs)
tagging_time = time.time()

# Decode predictions
init_predictions = torch.argmax(outputs.logits, dim=-1)[0]
split_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Convert input IDs to tokens
predicted_tags = [init_predictions[i].item() for i in range(len(split_tokens))]

# predicted_tags.reverse()
# split_tokens.reverse()
# temp_token = ""
# tokens = []
# predictions = []
# for prediction, token in zip(predicted_tags, tokens):
#     if token.startswith("##"):
#         temp_token = temp_token + token[2:]
#     else:
#         token = token + temp_token
#         tokens.append(token)
#         predictions.append(prediction)
#         temp_token = ""
# tokens.reverse()
# predictions.reverse()

tokens = split_tokens
predictions = predicted_tags
print(predictions)
print(len(predictions))
print(len(labels[0]))

total_token=0
total_correct=0
# Display the results
tokens, predictions, answers = assemble(tokens, predictions, labels[0])
for token, tag, answer in zip(tokens, predictions, answers):
    # Skip special tokens
    if token in ["[CLS]", "[SEP]"]:
        continue
    print(f"{token}: {ID_TO_NER.get(tag, 'UNKNOWN')}: {answer}")
    total_token += 1
    total_correct += tag == answer

percent_correct = total_correct/total_token

print(f"load time: {load_time - start_time}")
print(f"tokenizing time: {tokenize_time - load_time}")
print(f"tagging time: {tagging_time - tokenize_time}")
print(f"total time: {tagging_time - start_time}")
print(f"Percentage correct: {percent_correct}")