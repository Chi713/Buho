
import os
from scripts.utils import fragment_assembler as frag
from scripts.training.lemma import lemmatizer as lem

# Define paths
DATA_PATH = os.environ.get('BUHO_DATA_PATH')
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH')
NER_MODEL_PATH = os.path.join(MODEL_PATH, "ner")
POS_MODEL_PATH = os.path.join(MODEL_PATH, "pos")
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")

# Define your ID_TO_POS mapping
POS_TO_ID = {
    '-PAD-': 0, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5,
    'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11,
    'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16,
    'X': 17, '_': 18
}
ID_TO_POS = {v: k for k, v in POS_TO_ID.items()}
# Define your ID_TO_NER mapping
NER_TO_ID = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ID_TO_NER = {v: k for k, v in NER_TO_ID.items()}
max_sequence = 10

# str ='Located in the Pacific Ocean off the northeast coast of the Asian mainland, it is bordered on the west by the Sea of Japan and extends from the Sea of Okhotsk in the north to the East China Sea in the south. The Japanese archipelago consists of four major islands—Hokkaido, Honshu, Shikoku, and Kyushu—and thousands of smaller islands, covering 377,975 square kilometres'
str='Japón es un país insular de Asia Oriental ubicado en el noroeste del océano Pacífico. Limita con el mar del Japón al oeste y se extiende desde el mar de Ojotsk en el norte hasta el mar de la China Oriental y Taiwán en el sur. Su territorio comprende un archipiélago de 14 125 islas que cubren 377 978 km² sobre el denominado anillo de fuego del Pacífico; las cinco islas principales del país, de norte a sur, son Hokkaidō, Honshū, Shikoku, Kyūshū y Okinawa.'

def main():
    # Assuming `frag.max_sequence_splitter` splits the text into manageable chunks
    sentence = frag.max_sequence_splitter(str, max_sequence)

    # Run NER inference
    results_ner = frag.run_ner_inference(
        sentence,
        NER_MODEL_PATH,
        TOKENIZER_MODEL_PATH,
        ID_TO_NER,
        max_sequence
    )

    # Run POS inference
    results_pos = frag.run_pos_inference(
        sentence,
        POS_MODEL_PATH,
        TOKENIZER_MODEL_PATH,
        ID_TO_POS,
        max_sequence  
    )

    # Print combined results in the format: "original word: ner output: pos output"
    for ner_result, pos_result in zip(results_ner, results_pos):
        for (ner_token, ner_label), (pos_token, pos_label) in zip(ner_result, pos_result):
            # Assuming tokens from both results should match
            if ner_token not in ["[CLS]", "[SEP]"]:  # Skip special tokens if necessary
                lemma = lem.lemmatize(ner_token, pos_label)
                print(f"{ner_token}: {ner_label}: {pos_label}: {lemma}")

if __name__ == "__main__":
    main()
