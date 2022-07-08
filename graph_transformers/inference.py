import time
import numpy as np
from graph_transformers.modelling import (
    roberta,
    qagnn
)
from graph_transformers.preprocess import (
    statement,
    grounding,
    graph
)

import torch
from transformers import RobertaTokenizer


class Inference:
    def __init__(self,
                 inputs: dict,
                 use_lm: bool = True,
                 model_path: str = None,
                 ):
        self.inputs = inputs
        self.use_lm = use_lm
        self.model_path = model_path
        self.device = torch.device("cpu")

    def _load_lm(self):
        print('loading pre-trained LM...')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        lm_model = roberta.RobertaForMaskedLMwithLoss.from_pretrained('roberta-base')
        # lm_model.cuda();
        lm_model.eval()
        print('loading done')
        return lm_model, tokenizer

    def _prepare_data(self):
        statements = statement.convert_to_entailment(input=self.inputs)
        grounded = grounding.ground(
            statements,
            cpnet_vocab_path='./data/concept.txt',
            pattern_path='./data/matcher_patterns.json',
            num_processes=1
        )
        print(grounded)
        # get the graph
        if self.use_lm:
            model, tokenizer = self._load_lm()
            graph_adj = graph.generate_adj_data_from_grounded_concepts__use_LM(
                statements,
                grounded,
                cpnet_graph_path='./data/conceptnet.en.pruned.graph',
                cpnet_vocab_path='./data/concept.txt',
                model=model,
                tokenizer=tokenizer,
                num_processes=1
            )
        else:
            graph_adj = graph.generate_adj_data_from_grounded_concepts(
                grounded,
                cpnet_graph_path='./data/conceptnet.en.pruned.graph',
                cpnet_vocab_path='./data/concept.txt',
                num_processes=1
            )
        print(graph_adj)
        return statements, grounded, graph_adj

    def eval_detail(self):

        statements, grounded, graphs = self._prepare_data()

        assert self.model_path is not None
        print("check")
        # only for medqa data
        # cp_emb = [np.load("data/tzw.ent.npy")]
        # cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
        # concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
        # print('| num_concepts: {} |'.format(concept_num))
        #
        # print(concept_num, concept_dim)

        # load the model
        model_state_dict, old_args = torch.load(self.model_path, map_location=torch.device('cpu'))


        # dataset = qagnn.LM_QAGNN_DataLoader_inference(args, statements, graphs,
        #                                         args.test_statements, args.test_adj,
        #                                         batch_size=1, eval_batch_size=1,
        #                                         device=(self.device, self.device),
        #                                         model_name=old_args.encoder,
        #                                         max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
        #                                         is_inhouse=args.inhouse,
        #                                         inhouse_train_qids_path=args.inhouse_train_qids,
        #                                         subsample=args.subsample, use_cache=args.use_cache)

        # # create the model template
        model = qagnn.LM_QAGNN(
            old_args,
            old_args.encoder,
            k=old_args.k,
            n_ntype=4,
            n_etype=old_args.num_relation,
            # n_concept=concept_num,
            concept_dim=old_args.gnn_dim,
            # concept_in_dim=concept_dim,
            n_attention_head=old_args.att_head_num,
            fc_dim=old_args.fc_dim,
            n_fc_layer=old_args.fc_layer_num,
            p_emb=old_args.dropouti,
            p_gnn=old_args.dropoutg,
            p_fc=old_args.dropoutf,
            # pretrained_concept_emb=cp_emb,
            freeze_ent_emb=old_args.freeze_ent_emb,
            init_range=old_args.init_range,
            encoder_config={}
        )
        # print("after loading")
        #
        # # load the model
        # model.load_state_dict(model_state_dict)

        # cpu and gpu setting
        # if torch.cuda.device_count() >= 2 and args.cuda:
        #     device0 = torch.device("cuda:0")
        #     device1 = torch.device("cuda:1")
        # elif torch.cuda.device_count() == 1 and args.cuda:
        #     device0 = torch.device("cuda:0")
        #     device1 = torch.device("cuda:0")
        # else:
        #     device0 = torch.device("cpu")
        #     device1 = torch.device("cpu")

        # device0 = torch.device("cpu")
        # device1 = torch.device("cpu")
        # model.encoder.to(device0)
        # model.decoder.to(device1)
        # model.eval()

        # dataset = LM_QAGNN_DataLoader_inference(args, test_statement_json, test_graph,
        #                                         args.test_statements, args.test_adj,
        #                                         batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        #                                         device=(device0, device1),
        #                                         model_name=old_args.encoder,
        #                                         max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
        #                                         is_inhouse=args.inhouse,
        #                                         inhouse_train_qids_path=args.inhouse_train_qids,
        #                                         subsample=args.subsample, use_cache=args.use_cache)
        #
        # eval_set = dataset.test()
        #
        # for qids, labels, *input_data in eval_set:
        #     logits, _, = model(*input_data)
        #     predictions = logits.argmax(1)  # [bsize, ]
        #     print(f"The prediction of current input is : {predictions}")

    def evaluate(self):
        pass



if __name__ == '__main__':
    model_path = "../saved_models/csqa_model_hf3.4.0.pt"
    question = "There is a star at the center of what group of celestial bodies?"
    choices = ["hollywood", "skyline", "outer space", "constellation", "solar system"]
    input = {"question": question, "choices": choices}
    start = time.time()
    inf = Inference(inputs=input, use_lm=True, model_path=model_path)
    inf.eval_detail()
    end = time.time()
    print("Inference time: ", end-start)
