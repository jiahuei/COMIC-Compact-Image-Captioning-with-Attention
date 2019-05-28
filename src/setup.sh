#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/../common/coco_caption
bash get_stanford_models.sh

cd ${DIR}/../datasets/preprocessing
python coco_prepro.py --dataset_dir ''
#python insta_prepro.py --dataset_dir ''

