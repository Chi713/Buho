**BUHO_DIR** is working directory

**BUHO_CONLLU_DATA_PATH** is conllu file path

**$BUHO_DATA_PATH** is training/testing data path

**BUHO_MODEL_PATH** is models path


## Setup

create venv with required libraries

export environment variables
- for example .env file we put the dataset in the /cache folder

## Training

**Tokenizer**

In scripts/tokenizer

step 1: run add_tokens.py

step 2: run check_tokens.py

step 3: run parse_data.py

**POS**

in scripts/pos

step 1: run tokenizer_pos.py

step 2: run train_pos_model.py

## Example .env file

export BUHO_DIR= /path/to/work/dir
export HF_HOME=$BUHO_DIR/cache/huggingface_cache/
export BUHO_CONLLU_DATA_PATH=$BUHO_DIR/cache/UD_Spanish-AnCora
export BUHO_DATA_PATH=$BUHO_DIR/cache/data_cache
export BUHO_MODEL_PATH=$BUHO_DIR/models