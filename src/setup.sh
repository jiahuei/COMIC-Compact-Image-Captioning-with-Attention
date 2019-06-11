#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


cd ${DIR}/../common/coco_caption
printf "\nSetting up Stanford CoreNLP for SPICE ...\n"
bash get_stanford_models.sh


cd ${DIR}/../datasets/preprocessing
printf "\nRunning pre-processing script for MS-COCO ...\n"
python coco_prepro.py --dataset_dir ''

printf "\nRunning pre-processing script for InstaPIC-1.1M ...\n"
python insta_prepro.py --dataset_dir ''


cd ${DIR}/../common/scst
printf "\nRunning pre-processing script for SCST (MS-COCO) ...\n"
python prepro_ngrams.py --dataset_dir ''

printf "\nRunning pre-processing script for SCST (InstaPIC-1.1M) ...\n"
python prepro_ngrams.py --dataset_dir '' --dataset_file_pattern 'insta_{}_v25595_s15'


printf "\nSetup complete.\n"
