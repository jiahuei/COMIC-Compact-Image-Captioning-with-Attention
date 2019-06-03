#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}

### Training
# Default dataset directory
python train.py

# Custom dataset directory
python train.py  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'

# Word token, custom dataset directory, GPU 1
python train.py  \
	--token_type 'word'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'  \
	--gpu '1'  


### Inference
# Default dataset and checkpoint directories
python infer.py  \
	--infer_checkpoints_dir 'mscoco/radix_add_softmax_h8_tie_lstm_run_01'

# Custom dataset and checkpoint directories
python infer.py  \
	--infer_checkpoints_dir 'mscoco/word_add_softmax_h8_tie_lstm_run_01'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'   \
	--gpu '1'

