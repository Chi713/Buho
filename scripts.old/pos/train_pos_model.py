import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import classification_report
import numpy as np

# Paths to tokenized data
DATA_PATH = os.environ.get('BUHO_DATA_PATH')
MODELS_PATH = os.environ.get('BUHO_MODELS_PATH')
MODEL_NAME = os.environ.get('BUHO_MODEL_NAME')

MODEL_PATH =  os.path.join(MODELS_PATH, MODEL_NAME)
TRAIN_PATH = os.path.join(DATA_PATH, "train_inputs_pos.pt")
DEV_PATH = os.path.join(DATA_PATH, "dev_inputs_pos.pt")
TEST_PATH = os.path.join(DATA_PATH, "test_inputs_pos.pt")

# POS Mapping
POS_TO_ID = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5, 'INTJ': 6, 'NOUN': 7, 'NUM': 8, 
             'PART': 9, 'PRON': 10, 'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'SYM': 14, 'VERB': 15, 'X': 16, '_': 17}
ID_TO_POS = {v: k for k, v in POS_TO_ID.items()}

# Dataset Class
class POSTaggingDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "labels": self.data["labels"][idx],
        }

# Load Datasets
train_dataset = POSTaggingDataset(TRAIN_PATH)
dev_dataset = POSTaggingDataset(DEV_PATH)
test_dataset = POSTaggingDataset(TEST_PATH)

# Load BERT Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, num_labels=len(POS_TO_ID))

# Data Collator (Handles Padding Dynamically During Training)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Training Arguments
training_args = TrainingArguments(
    output_dir="../results",
    eval_strategy="epoch",  # Replace evaluation_strategy with eval_strategy
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    save_strategy="epoch",
    gradient_accumulation_steps=2,
    fp16=True,
    logging_steps=10,
)


# Trainer Object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train Model
trainer.train()

# Evaluate Model
predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=2)

# Align Predictions and Labels
y_pred = []
y_true = []

for pred, label in zip(predictions, labels):
    for p, l in zip(pred, label):
        if l != -100:  # Ignore padding
            y_pred.append(p)
            y_true.append(l)
            

# Dynamically include only the labels present in the test set
unique_labels = sorted(set(y_true))  # Ensure unique labels in the test set
filtered_target_names = [ID_TO_POS[i] for i in unique_labels]

print(classification_report(
    y_true,
    y_pred,
    target_names=filtered_target_names,
    labels=unique_labels  # Use only the labels present in the test set
))

# Save Model
model.save_pretrained(MODEL_PATH)
### why are we saving the tokenizer again?? it wasn't trained here
# tokenizer.save_pretrained(os.path.join(MODELS_PATH, "buho_pos_model_trained"))
print("Model and tokenizer saved to 'buho_pos_model_trained'.")
