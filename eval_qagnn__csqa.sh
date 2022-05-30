#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
dt=`date '+%Y%m%d_%H%M%S'`


dataset="csqa"
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
python3 -u qagnn.py --dataset $dataset \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements data/${dataset}/statement/train.statement.jsonl \
      --dev_statements   data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir saved_models \
      --mode eval_detail \
      --load_model_path saved_models/csqa_model_hf3.4.0.pt \
      $args
#{"id": "90b30172e645ff91f7171a048582eb8b", "question": {"question_concept": "townhouse", "choices": [{"label": "A", "text": "suburban development"}, {"label": "B", "text": "apartment building"}, {"label": "C", "text": "bus stop"}, {"label": "D", "text": "michigan"}, {"label": "E", "text": "suburbs"}], "stem": "The townhouse was a hard sell for the realtor, it was right next to a high rise what?"}, "statements": [{"label": true, "statement": "The townhouse was a hard sell for the realtor, it was right next to a high rise suburban development."}, {"label": false, "statement": "The townhouse was a hard sell for the realtor, it was right next to a high rise apartment building."}, {"label": false, "statement": "The townhouse was a hard sell for the realtor, it was right next to a high rise bus stop."}, {"label": false, "statement": "The townhouse was a hard sell for the realtor, it was right next to a high rise michigan."}, {"label": false, "statement": "The townhouse was a hard sell for the realtor, it was right next to a high rise suburbs."}]}
