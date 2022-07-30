import time
import numpy as np
from tqdm import tqdm
import json
from itertools import product
from graph_transformers.modelling import (
    roberta,
    qagnn
)
from graph_transformers.preprocess import (
    statement,
    grounding,
    graph,
    kgapi
)

import torch
from transformers import (
    AutoTokenizer,
    RobertaTokenizer,
    OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
)


MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}


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
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        lm_model = roberta.RobertaForMaskedLMwithLoss.from_pretrained('roberta-base')
        # lm_model.cuda();
        lm_model.eval()
        print('loading done')
        return lm_model, tokenizer

    def _prepare_data(self):
        statements = statement.convert_to_entailment(input=self.inputs)
        print(statements)
        grounded = grounding.ground(
            statements,
            cpnet_vocab_path='./data/concept.txt',
            pattern_path='./data/matcher_patterns.json',
            num_processes=1
        )
        # print(grounded)
        # get the graph
        if self.use_lm:
            model, tokenizer = self._load_lm()
            # enable the api with
            #graph_adj = graph.generate_adj_data_from_grounded_concepts__use_LM_api()
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
        # print(graph_adj)
        return statements, grounded, graph_adj

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    def convert_features_to_tensors(self,features):

        all_input_ids = torch.tensor(self.select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(self.select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(self.select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(self.select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    def select_field(self, features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]


    def _convert_examples_to_features(
            self,
            examples,
            max_seq_length,
            tokenizer,
            cls_token_at_end=False,
            cls_token='[CLS]',
            cls_token_segment_id=1,
            sep_token='[SEP]',
            sequence_a_segment_id=0,
            sequence_b_segment_id=1,
            sep_token_extra=False,
            pad_token_segment_id=0,
            pad_on_left=False,
            pad_token=0,
            mask_padding_with_zero=True
    ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to
                the CLS token (0 for BERT, 2 for XLNet)
        """

        class InputFeatures(object):

            # def __init__(self, input_ids, input_mask, segment_ids, output_mask, label):
            #
            #     self.input_ids = input_ids,
            #     self.input_mask = input_mask,
            #     self.segment_ids = segment_ids,
            #     self.output_mask = output_mask,
            #     self.label = label
            def __init__(self,  choices_features, label):

                self.choices_features = [
                    {
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'segment_ids': segment_ids,
                        'output_mask': output_mask,
                    }
                    for _, input_ids, input_mask, segment_ids, output_mask in choices_features
                ]
                self.label = label

        question = examples["question"]
        endings = examples["choices"]
        contexts = [question] * len(endings)
        label_list = list(range(len(endings)))

        labels = ord(examples["answerKey"]) - ord("A") if 'answerKey' in examples else 0
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        # for ex_index, example in enumerate(tqdm(examples)):
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(contexts, endings)):
            tokens_a = tokenizer.tokenize(context)
            tokens_b = tokenizer.tokenize(question + " " + ending)

            special_tokens_count = 4 if sep_token_extra else 3
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.

            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
            output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                output_mask = ([1] * padding_length) + output_mask

                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(output_mask) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label = label_map[labels]

            # convert to tensors
            # input_ids = torch.tensor(input_ids, dtype=torch.long)
            # input_mask = torch.tensor(input_mask, dtype=torch.long)
            # segment_ids = torch.tensor(segment_ids, dtype=torch.long)
            # output_mask = torch.tensor(output_mask, dtype=torch.bool)
            # label = torch.tensor([label], dtype=torch.long)



            choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
        features.append(InputFeatures(
                choices_features=choices_features,
                label=label
            ))
        print("check point")

        return features

    def load_sparse_adj_data_with_contextnode(self,adj_concept_pairs , max_node_num, num_choice, ):
        n_samples = len(adj_concept_pairs)  # this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)  # default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)

        adj_lengths_ori = adj_lengths.clone()
        for idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
            adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data[
                'cid2score']
            # adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            # concepts: np.array(num_nodes, ), where entry is concept id
            # qm: np.array(num_nodes, ), where entry is True/False
            # am: np.array(num_nodes, ), where entry is True/False
            concepts = np.array(list(set(concepts))) ## TODO: should be removec after fixing the api concept2id_api ()
            assert len(concepts) == len(set(concepts))
            qam = qm | am
            # sanity check: should be T,..,T,F,F,..F
            assert qam[0] == True
            F_start = False
            for TF in qam:
                if TF == False:
                    F_start = True
                else:
                    assert F_start == False
            num_concept = min(len(concepts),
                              max_node_num - 1) + 1  # this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            # Prepare nodes
            concepts = concepts[:num_concept - 1]
            concept_ids[idx, 1:num_concept] = torch.tensor(
                concepts + 1)  # To accomodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = 0  # this is the "concept_id" for contextnode

            # Prepare node scores
            if (cid2score is not None):
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1
                    assert _cid in cid2score
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            # Prepare node types
            node_type_ids[idx, 0] = 3  # contextnode
            node_type_ids[idx, 1:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept - 1]] = 0
            node_type_ids[idx, 1:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept - 1]] = 1

            # Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            # Prepare edges
            i += 2;
            j += 1;
            k += 1  # **** increment coordinate by 1, rel_id by 2 ****
            extra_i, extra_j, extra_k = [], [], []
            for _coord, q_tf in enumerate(qm):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if q_tf:
                    extra_i.append(0)  # rel from contextnode to question concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # question concept coordinate
            for _coord, a_tf in enumerate(am):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if a_tf:
                    extra_i.append(1)  # rel from contextnode to answer concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # answer concept coordinate

            half_n_rel += 2  # should be 19 now
            if len(extra_i) > 0:
                i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                k = torch.cat([k, torch.tensor(extra_k)], dim=0)
            ########################

            mask = (j < max_node_num) & (k < max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j),
                                                                                         0)  # add inverse relations
            edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
            edge_type.append(i)  # each entry is [E, ]

        # with open(cache_path, 'wb') as f:
        #     pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel], f)

        ori_adj_mean = adj_lengths_ori.float().mean().item()
        ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean) ** 2).mean().item())
        print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma,
                                                                                 adj_lengths.float().mean().item()) +
              ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
              ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                          (node_type_ids == 1).float().sum(1).mean().item()))

        edge_index = list(map(list, zip(*(iter(
            edge_index),) * num_choice)))  # list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
        edge_type = list(map(list, zip(*(iter(
            edge_type),) * num_choice)))  # list of size (n_questions, n_choices), where each entry is tensor[E, ]

        concept_ids, node_type_ids, node_scores, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in
                                                                (concept_ids, node_type_ids, node_scores, adj_lengths)]

        # concept_ids: (n_questions, num_choice, max_node_num)
        # node_type_ids: (n_questions, num_choice, max_node_num)
        # node_scores: (n_questions, num_choice, max_node_num)
        # adj_lengths: (n_questions,　num_choice)
        return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type)  # , half_n_rel * 2 + 1


    def _predict(self):

        statements, grounded, graphs = self._prepare_data()
        self.grounded = grounded
        self.graphs = graphs
        model_type = "roberta"
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        assert self.model_path is not None

        # only for medqa data
        cp_emb = [np.load("data/tzw.ent.npy")]
        cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
        concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
        print('| num_concepts: {} |'.format(concept_num))

        print(concept_num, concept_dim)

        # load the model
        model_state_dict, old_args = torch.load(self.model_path, map_location=torch.device('cpu'))

        # # create the model template
        model = qagnn.LM_QAGNN(
            # old_args,
            # old_args.encoder,
            # k=old_args.k,
            # n_ntype=4,
            # n_etype=old_args.num_relation,
            # n_concept=concept_num,
            # concept_dim=old_args.gnn_dim,
            # concept_in_dim=concept_dim,
            # n_attention_head=old_args.att_head_num,
            # fc_dim=old_args.fc_dim,
            # n_fc_layer=old_args.fc_layer_num,
            # p_emb=old_args.dropouti,
            # p_gnn=old_args.dropoutg,
            # p_fc=old_args.dropoutf,
            # pretrained_concept_emb=cp_emb,
            # freeze_ent_emb=old_args.freeze_ent_emb,
            # init_range=old_args.init_range,
            # encoder_config={}
            old_args, old_args.encoder, k=old_args.k, n_ntype=4, n_etype=old_args.num_relation, n_concept=concept_num,
            concept_dim=old_args.gnn_dim,
            concept_in_dim=concept_dim,
            n_attention_head=old_args.att_head_num, fc_dim=old_args.fc_dim, n_fc_layer=old_args.fc_layer_num,
            p_emb=old_args.dropouti, p_gnn=old_args.dropoutg, p_fc=old_args.dropoutf,
            pretrained_concept_emb=cp_emb, freeze_ent_emb=old_args.freeze_ent_emb,
            init_range=old_args.init_range,
            encoder_config={}
        )
        print("after loading")

        # load the model
        model.load_state_dict(model_state_dict,False)

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

        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
        model.encoder.to(device0)
        model.decoder.to(device1)
        model.eval()

        features = self._convert_examples_to_features(
            examples=statements,
            max_seq_length=128,
            tokenizer=tokenizer,
            cls_token_at_end=bool(model_type in ['xlnet']),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ['roberta', 'albert']),
            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
            sequence_b_segment_id=0 if model_type in ['roberta', 'albert'] else 1
        )
        self.num_choice = len(statements['choices'])
        *data_tensors, all_label = self.convert_features_to_tensors(features)

        *test_decoder_data, test_adj_data = self.load_sparse_adj_data_with_contextnode(graphs, max_node_num=200,
                                                                                            num_choice=self.num_choice, )

        # print(features)
        # *input_data, labels = features
        input_data = [*data_tensors,*test_decoder_data, *test_adj_data]

        # logits, attn, = model(*input_data )
        logits, attn, concept_ids, node_type_ids, edge_index_orig, edge_type_orig= model(*input_data,detail= True )

        self.attn = attn
        self.info = input_data
        predictions = logits.argmax(1)  # [bsize, ]
        self.prediction = predictions

        print(f"The prediction of current input is : {predictions}")



    def _get_attn(self):


        info = dict()
        info['node_ids'] = self.info[4].squeeze() # (5,200)
        torch.save(self.info[4].squeeze(),'node_ids.pt')
        info['scores'] = self.info[6].squeeze() # (5,200)
        torch.save(self.info[6].squeeze(), 'scores.pt')
        attn_h1 = self.attn[:self.num_choice]
        attn_h2 = self.attn[self.num_choice:]
        info['attn'] = (attn_h1+attn_h2)/2 # (5,200)

        torch.save((attn_h1+attn_h2)/2, 'attn.pt')
        # torch.save(attn_h1 , 'attn_h1.pt')
        # torch.save(attn_h2, 'attn_h2.pt')
        info['edge_index'] = self.info[8] # list of 5 lists
        info['edge_type'] = self.info[9] # list of 5 lists

    def _get_info(self):
        # info = dict()
        # info['node_ids'] = self.info[4].squeeze() # (5,200)
        # torch.save(self.info[4].squeeze(),'node_ids.pt')
        # info['scores'] = self.info[6].squeeze() # (5,200)
        # torch.save(self.info[6].squeeze(), 'scores.pt')
        # attn_h1 = self.attn[:self.num_choice]
        # attn_h2 = self.attn[self.num_choice:]
        # info['attn'] = (attn_h1+attn_h2)/2 # (5,200)
        #
        # torch.save((attn_h1+attn_h2)/2, 'attn.pt')
        # torch.save(attn_h1 , 'attn_h1.pt')
        # torch.save(attn_h2, 'attn_h2.pt')
        # info['edge_index'] = self.info[8] # list of 5 lists
        # info['edge_type'] = self.info[9] # list of 5 lists

        def node_info(node_id:int,score_map:dict,grounded:dict):
            node = dict()
            node['id']=node_id
            node['name'] = kgapi.id2concept_api(np.abs(node_id))
            node['description']=""
            node['q_node']= False
            node['ans_node'] = False
            if node['name'] in grounded['qc'] :
                node['q_node'] = True
            elif node['name'] in grounded['ac']:
                node['ans_node']= True
            node['width']=score_map[node_id]
            return node

        def edge_info(node_ids:list):
            pair_list = [list(i) for i in list(product(node_ids, node_ids))]
            edges = kgapi.get_edges_by_node_pairs(pair_list)

            re = dict()
            for i in edges.items():
                tmp_dict = dict()
                tmp_dict['source'] = kgapi.nid_to_int(i[1]['in_id'])
                tmp_dict['target'] = kgapi.nid_to_int(i[1]['out_id'])
                tmp_dict['weight'] = i[1]['weight']
                tmp_dict['label'] = i[1]['name']
                re[i[0]] = tmp_dict
            return re

        info =dict()
        info['nodes'] = dict()
        info['scores'] = dict()
        info['edges'] = dict()
        for i in range(len(self.graphs)):
            score = self.graphs[i]['cid2score']
            keys = [int(k) for k in score.keys()]
            values = [float(v) for v in score.values()]
            # score = json.dumps(dict(zip(keys, values)),indent=4)
            score = dict(zip(keys, values))
            info['nodes']['statement_' + str(i)] = dict()
            for k in keys:
                info['nodes']['statement_'+str(i)][str(k)] = node_info(k,score,self.grounded[i])

            info['scores']['statement_'+str(i)] = score

            info['edges']['statement_' + str(i)] = dict()
            info['edges']['statement_' + str(i)] = edge_info(keys)

        info = json.dumps(info,indent=4)
        with open('info_demo.json', 'w') as f:
            f.write(info)

        return info


if __name__ == '__main__':
    model_path = "saved_models/csqa_model_hf3.4.0.pt"
    # question = "There is a star at the center of what group of celestial bodies?"
    # choices = ["hollywood", "skyline", "outer space",""constellation", "solar system"]
    # question = "What is the color of an apple?"
    # choices = ["black","red","blue","orange","yellow"]

    # question = "Crabs live in what sort of enviroment?"
    # choices = ["saltwater","galapagos","fish market"]

    # question = "Where would you find a basement that can be accesssed with an elevator?"
    # choices = ["closet","church","office building"]

    question = "A revolving door is convenient for two direction travel, but also serves as a security measure at what?"
    choices = ["bank", "library", "department store", "mall", "new york"]


    input = {"question": question, "choices": choices}
    start = time.time()
    inf = Inference(inputs=input, use_lm=True, model_path=model_path)
    inf._predict()
    info =inf._get_info()
    # info = inf._get_attn()
    # print(info)


    # print(inf._get_node_score())
    # print(inf.node_scores)
    end = time.time()
    print("Inference time: ", end-start)