import os
import pyconll
from transformers import AutoTokenizer


CONLLU_PATH = os.environ.get('BUHO_CONLLU_DATA_PATH')
DATA_PATH = os.environ.get('BUHO_DATA_PATH')

MODEL_PATH = os.environ.get('BUHO_MODEL_PATH')
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")
PRETRAINED_MODEL = os.environ.get('PRETRAINED_MODEL')

FILES = {
    "train": "es_ancora-ud-train.conllu",
    "dev": "es_ancora-ud-dev.conllu",
    "test": "es_ancora-ud-test.conllu"
}
def get_training_corpus(dataset):
    return (
        dataset[i : i + 1000]
        for i in range(0, len(dataset), 1000)
    )


train_file = os.path.join(CONLLU_PATH, FILES["train"])
conll = pyconll.load_from_file(train_file)
sentences = [sentence.text for sentence in conll]
parsed_sentences = [[token.form for token in sentence] for sentence in conll]
# sentence_trees = [sentence.to_tree() for sentence in conll]
# print(sentence_trees[0].data.form)
output_sentences = [sentence + '[' + ','.join(parsed_sentence) + ']' for sentence, parsed_sentence in zip(sentences, parsed_sentences)]

training_corpus = get_training_corpus(output_sentences)

old_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
tokenizer.save_pretrained(TOKENIZER_MODEL_PATH)
print(len(old_tokenizer))