#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}

### Training
# Default
python train.py

# Custom MS-COCO directory
python train.py  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'

# Word token, custom MS-COCO directory, GPU 1
python train.py  \
	--token_type 'word'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'  \
	--gpu '1'  

# InstaPIC
python train.py  \
    --dataset_file_pattern 'insta_{}_v25595_s15'

# Custom InstaPIC directory
python train.py  \
    --dataset_file_pattern 'insta_{}_v25595_s15'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/InstaPIC'


### Inference
# Default dataset and checkpoint directories (MSCOCO, COMIC-256)
python infer.py

# Custom dataset and checkpoint directories
python infer.py  \
	--infer_checkpoints_dir 'mscoco/word_add_softmax_h8_tie_lstm_run_01'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'   \
	--gpu '1'

# InstaPIC
python infer.py  \
	--infer_checkpoints_dir 'insta/word_add_softmax_h8_tie_lstm_run_01'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/InstaPIC'  \
	--annotations_file 'insta_testval_raw.json'

# Custom InstaPIC directory
python infer.py  \
	--infer_checkpoints_dir 'insta/word_add_softmax_h8_tie_lstm_run_01'  \
	--dataset_dir '/home/jiahuei/Documents/3_Datasets/InstaPIC'  \
	--annotations_file 'insta_testval_raw.json'



