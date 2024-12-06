import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

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
            temp_token = temp_token + token[2:]
        else:
            token = token + temp_token
            assembled_tokens.append(token)
            new_predictions.append(prediction)
            new_answers.append(answer)
            temp_token = ""

    print(assembled_tokens, new_predictions, new_answers)
    return assembled_tokens, new_predictions, new_answers

def max_sequence_splitter(input_str, max_length=256):
    # Split the input string into individual words
    words = input_str.split()

    output = []
    # Iterate over the words in steps of max_length
    for i in range(0, len(words), max_length):
        chunk = words[i:i+max_length]
        # Convert the chunk back into a single string
        chunk_str = " ".join(chunk)
        output.append(chunk_str)

    return output
import os
import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

def assemble(tokens: list[str], predictions: list[int], answers: list[int]):
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
            temp_token = ""
    assembled_tokens.reverse()
    new_predictions.reverse()
    new_answers.reverse()

    return assembled_tokens, new_predictions, new_answers


def run_ner_inference(
    sentences: list[str],
    model_path: str,
    tokenizer_path: str,
    id_to_ner: dict[int, str],
    max_sequence: int = 256
):
    """
    Runs inference on a list of sentences, each represented as a string.

    Args:
        sentences (list[str]): A list of sentences, each containing multiple words.
        model_path (str): Path to the pretrained token classification model.
        tokenizer_path (str): Path to the tokenizer.
        id_to_ner (dict[int, str]): Dictionary mapping label IDs to label strings.
        max_sequence (int): Maximum number of words per sentence.

    Returns:
        list[list[tuple[str, str]]]: A list of results, where each result corresponds
        to one of the input sentences. Each result is a list of (token, label) tuples.
    """
    # Load model and tokenizer (ideally these should be loaded once outside the function)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model.eval()

    all_results = []
    for sentence in sentences:
        # Split the sentence into words and truncate
        words = sentence.split()
        truncated_words = words[:max_sequence]

        # Tokenize with is_split_into_words=True
        inputs = tokenizer(
            truncated_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_sequence
        )

        # Create dummy labels to allow assembly (not used for loss or accuracy)
        word_ids = inputs.word_ids()
        labels = [0 if idx is not None else 0 for idx in word_ids]

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predictions and convert IDs to tokens
        init_predictions = torch.argmax(outputs.logits, dim=-1)[0]
        split_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [init_predictions[i].item() for i in range(len(split_tokens))]

        # Assemble tokens and predictions
        assembled_tokens, assembled_predictions, _ = assemble(split_tokens, predicted_tags, labels)

        # Convert predicted tag IDs to their string labels
        predicted_labels = [id_to_ner.get(tag_id, "UNKNOWN") for tag_id in assembled_predictions]

        # Pair tokens with predicted labels
        sentence_results = list(zip(assembled_tokens, predicted_labels))

        all_results.append(sentence_results)

    return all_results

def run_pos_inference(
    sentences: list[str],
    model_path: str,
    tokenizer_path: str,
    id_to_pos: dict[int, str],
    max_sequence: int = 128
):
    """
    Runs POS inference on a list of sentences, each represented as a string.

    Args:
        sentences (list[str]): A list of sentences, each containing multiple words.
        model_path (str): Path to the pretrained token classification model.
        tokenizer_path (str): Path to the tokenizer.
        id_to_pos (dict[int, str]): Dictionary mapping label IDs to POS label strings.
        max_sequence (int): Maximum number of words per sentence.

    Returns:
        list[list[tuple[str, str]]]: A list of results, where each result corresponds
        to one of the input sentences. Each result is a list of (token, label) tuples.
    """
    # Load model and tokenizer (ideally these should be loaded once outside the function)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model.eval()

    all_results = []

    for sentence in sentences:
        # Split the sentence into words and truncate
        words = sentence.split()
        truncated_words = words[:max_sequence]

        # Tokenize with is_split_into_words=True
        inputs = tokenizer(
            truncated_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_sequence
        )

        # Create dummy labels to allow assembly (not used for loss or accuracy)
        word_ids = inputs.word_ids()
        labels = [0 if idx is not None else 0 for idx in word_ids]

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predictions and convert IDs to tokens
        init_predictions = torch.argmax(outputs.logits, dim=-1)[0]
        split_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [init_predictions[i].item() for i in range(len(split_tokens))]

        # Assemble tokens and predictions
        assembled_tokens, assembled_predictions, _ = assemble(split_tokens, predicted_tags, labels)

        # Convert predicted tag IDs to their string labels
        predicted_labels = [id_to_pos.get(tag_id, "UNKNOWN") for tag_id in assembled_predictions]

        # Pair tokens with predicted labels
        sentence_results = list(zip(assembled_tokens, predicted_labels))

        all_results.append(sentence_results)

    return all_results