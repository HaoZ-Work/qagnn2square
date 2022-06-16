#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
dt=`date '+%Y%m%d_%H%M%S'`



dataset="mytest"
model='roberta-large'
shift
shift
args=$@


echo "******************************"
echo "dataset: $dataset"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref

###### Eval ######
python3 -u inference.py --mode eval_detail -ih False --load_model_path saved_models/csqa_model_hf3.4.0.pt $args
