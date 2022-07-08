cd data/cpnet/
wget https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy
cd ../../

mkdir saved_models
cd saved_models
wget https://nlp.stanford.edu/projects/myasu/QAGNN/models/csqa_model_hf3.4.0.pt

#wget https://nlp.stanford.edu/projects/myasu/QAGNN/models/obqa_model_hf3.4.0.pt
#wget https://nlp.stanford.edu/projects/myasu/QAGNN/models/medqa_usmle_model_hf3.4.0.pt