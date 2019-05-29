#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}

python train.py  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'


python train.py  \
	--token_type 'word'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'  \
	--gpu '1'  

python infer.py  \
	--infer_checkpoints_dir '/home/jiahuei/Dropbox/@_PhD/Codes/COMIC/experiments/mscoco/radix_add_softmax_h8_tie_lstm_run_01'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'

python infer.py  \
	--infer_checkpoints_dir '/home/jiahuei/Dropbox/@_PhD/Codes/COMIC/experiments/mscoco/word_add_softmax_h8_tie_lstm_run_01'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'   \
	--gpu '1'

