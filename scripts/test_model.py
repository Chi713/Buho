from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Define ID_TO_POS mapping
ID_TO_POS = {
    0: 'ADJ', 1: 'ADP', 2: 'ADV', 3: 'AUX', 4: 'CCONJ',
    5: 'DET', 6: 'INTJ', 7: 'NOUN', 8: 'NUM', 9: 'PART',
    10: 'PRON', 11: 'PROPN', 12: 'PUNCT', 13: 'SCONJ',
    14: 'SYM', 15: 'VERB', 16: 'X', 17: '_'
}

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("pos_tagging_model")
tokenizer = AutoTokenizer.from_pretrained("pos_tagging_model")
model.eval()

# Test sentence
sentence = "La Unión Europea fue fundada en 1993."

# Tokenize input without `is_split_into_words`
inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

# Run inference
outputs = model(**inputs)

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
