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
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH')
TRAIN_PATH = os.path.join(DATA_PATH, "train_inputs_lemmas.pt")
DEV_PATH = os.path.join(DATA_PATH, "dev_inputs_lemmas.pt")
TEST_PATH = os.path.join(DATA_PATH, "test_inputs_lemmas.pt")

# Build Lemma Vocabulary
def build_lemma_vocab(train_path, dev_path, test_path):
    def get_unique_lemmas(dataset_path):
        dataset = torch.load(dataset_path)
        unique_lemmas = set()
        for lemmas in dataset["lemmas"]:
            for lemma_id in lemmas:
                if lemma_id != -100:
                    unique_lemmas.add(lemma_id)
        return unique_lemmas

    train_lemmas = get_unique_lemmas(train_path)
    dev_lemmas = get_unique_lemmas(dev_path)
    test_lemmas = get_unique_lemmas(test_path)

    all_lemmas = train_lemmas.union(dev_lemmas).union(test_lemmas)
    return sorted(all_lemmas)

lemma_vocab = build_lemma_vocab(TRAIN_PATH, DEV_PATH, TEST_PATH)
lemma_id_to_idx = {lemma_id: idx for idx, lemma_id in enumerate(lemma_vocab)}
idx_to_lemma_id = {idx: lemma_id for lemma_id, idx in lemma_id_to_idx.items()}
num_labels = len(lemma_vocab)
print(f"Number of unique lemmas: {num_labels}")

# Dataset Class
class LemmaTaggingDataset(Dataset):
    def __init__(self, data_path, lemma_vocab):
        self.data = torch.load(data_path)
        self.lemma_id_to_idx = {lemma_id: idx for idx, lemma_id in enumerate(lemma_vocab)}

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        labels = []
        for lemma in self.data["lemmas"][idx]:
            if lemma != -100:
                if lemma not in self.lemma_id_to_idx:
                    raise ValueError(f"Invalid lemma ID: {lemma}")
                labels.append(self.lemma_id_to_idx[lemma])
            else:
                labels.append(-100)
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "labels": torch.tensor(labels),
        }

# Load Datasets
train_dataset = LemmaTaggingDataset(TRAIN_PATH, lemma_vocab)
dev_dataset = LemmaTaggingDataset(DEV_PATH, lemma_vocab)
test_dataset = LemmaTaggingDataset(TEST_PATH, lemma_vocab)

# Step 3: Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModelForTokenClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased", num_labels=num_labels
)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing to save memory

# Step 4: Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Step 5: Training Arguments
training_args = TrainingArguments(
    output_dir="../results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # batch size
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    save_strategy="epoch",
    gradient_accumulation_steps=1,
    fp16=False,  # Disable mixed precision for debugging
    logging_steps=10,
)

# Step 6: Trainer Object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
)

# Step 7: Train Model
trainer.train()

# Step 8: Evaluate Model
torch.cuda.empty_cache()
predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=2)

# Step 9: Align Predictions and Labels
y_pred = []
y_true = []

for pred, label in zip(predictions, labels):
    for p, l in zip(pred, label):
        if l != -100:  # Ignore padding
            y_pred.append(idx_to_lemma_id[p])  # Map back to lemma IDs
            y_true.append(idx_to_lemma_id[l])

# Step 10: Classification Report
unique_labels = sorted(set(y_true))
filtered_target_names = [tokenizer.convert_ids_to_tokens(lemma_id) for lemma_id in unique_labels]

print(classification_report(
    y_true,
    y_pred,
    target_names=filtered_target_names,
    labels=unique_labels
))

# Step 11: Save Model
model.save_pretrained(os.path.join(MODEL_PATH, "lemma_tagging_model"))
tokenizer.save_pretrained(os.path.join(MODEL_PATH, "lemma_tagging_model"))
print("Model and tokenizer saved to 'lemma_tagging_model'.")
