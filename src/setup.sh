#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}/../datasets/preprocessing

python coco_prepro.py --dataset_dir ''
#python insta_prepro.py --dataset_dir ''

