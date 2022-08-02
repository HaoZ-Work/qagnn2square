import time
import os
from collections import OrderedDict
import json
from typing import List
import numpy as np
from tqdm import tqdm
import networkx as nx
import spacy
from spacy.matcher import Matcher
from itertools import product
import pprint
import operator


from graph_transformers_v2.modelling import (
    roberta_model,
    qagnn_model
)
from graph_transformers_v2.preprocess import (
    statement,
    grounding,
    graph,
    kgapi
)

import torch
from transformers import (
    AutoTokenizer,
    OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    logging
)
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}


DATA_DIR = "/work/home/kurse/kurs00056/hz53kahe/gnn/data/"

CPNET_VOCAB = DATA_DIR+"concept.txt"
CPNET_PATH = DATA_DIR+"conceptnet.en.pruned.graph"
PATTERN_PATH = DATA_DIR+"matcher_patterns.json"
MODEL_PATH = DATA_DIR+"csqa_model_hf3.4.0.pt"

LM_MODEL = "roberta-large"
MAX_NODE_NUM = 200
NUM_CHOICE = 5

nlp = None
matcher = None


class Inference:
    def __init__(
            self,
             inputs: List[List],
             model_path: str = None,
        ):
        self.inputs = inputs
        self.model_path = model_path
        self.device = torch.device("cpu")
        start = time.time()
        # load matcher
        self.nlp, self.matcher = self._load_matcher()
        # load DS
        self._load_resources(CPNET_VOCAB)
        self._load_cpnet(CPNET_PATH)
        # load lm model on init
        self._load_lm()
        self._load_qagnn()
        end = time.time()
        print(f"LOADING modules takes {end - start}")

    def _load_matcher(self):
        nlp = spacy.load(
            'en_core_web_sm',
            disable=['ner', 'parser', 'textcat'])
        # nlp.add_pipe('sentencizer')
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        with open(PATTERN_PATH, "r", encoding="utf8") as fin:
            all_patterns = json.load(fin)
        matcher = Matcher(nlp.vocab)
        for concept, pattern in all_patterns.items():
            matcher.add(concept, [pattern])
        print("Loaded matcher!!!")
        return nlp, matcher

    def _load_resources(self, cpnet_vocab_path):
        global concept2id, id2concept
        with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
            id2concept = [w.strip() for w in fin]
        concept2id = {w: i for i, w in enumerate(id2concept)}

    def _load_cpnet(self, cpnet_graph_path):
        global cpnet, cpnet_simple
        cpnet = nx.read_gpickle(cpnet_graph_path)  # Multigraph class
        cpnet_simple = nx.Graph()  # Graph class
        for u, v, data in cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        # print(cpnet_simple.nodes)
        print("Loaded conceptnet!!!")

    def _load_lm(self):

        self.tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
        self.lm_model = roberta_model.RobertaForMaskedLMwithLoss.from_pretrained(LM_MODEL)
        # lm_model.cuda();
        self.lm_model.eval()
        print('Loaded LM!!!')

    def _load_qagnn(self):
        cp_emb = [np.load(DATA_DIR+"tzw.ent.npy")]
        cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
        concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

        # load the model
        model_state_dict, model_args = torch.load(
            self.model_path,
            map_location=self.device)

        # create the model template
        self.model = qagnn_model.LM_QAGNN(
            model_args,
            model_args.encoder,
            k=model_args.k,
            n_ntype=4,
            n_etype=model_args.num_relation,
            n_concept=concept_num,
            concept_dim=model_args.gnn_dim,
            concept_in_dim=concept_dim,
            n_attention_head=model_args.att_head_num,
            fc_dim=model_args.fc_dim,
            n_fc_layer=model_args.fc_layer_num,
            p_emb=model_args.dropouti,
            p_gnn=model_args.dropoutg,
            p_fc=model_args.dropoutf,
            pretrained_concept_emb=cp_emb,
            freeze_ent_emb=model_args.freeze_ent_emb,
            init_range=model_args.init_range,
            encoder_config={}
        )

        # load the model
        self.model.load_state_dict(model_state_dict, False)
        self.model.encoder.to(self.device)
        self.model.decoder.to(self.device)
        self.model.eval()
        print("Loaded QaGNN!!!")

    def _prepare_data(self):
        global id2concept, concept2id, cpnet, cpnet_simple
        statements = statement.convert_to_entailment(input=self.inputs)
        # start = time.time()
        grounded = grounding.ground(
            statements,
            cpnet_vocab=id2concept,
            _nlp=self.nlp,
            _matcher=self.matcher,
            num_processes=8
        )
        # end = time.time()
        # print(f"grounding takes {end - start}")

        graph_adj = graph.generate_adj_data_from_grounded_concepts__use_LM(
            # qa_data,
            statements,
            grounded,
            concept2id=concept2id,
            _cpnet_vocab=id2concept,
            _cpnet=cpnet,
            _cpnet_simple=cpnet_simple,
            model=self.lm_model,
            tokenizer=self.tokenizer,
            num_processes=16
        )
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

        all_input_ids = torch.tensor(
            self.select_field(features, 'input_ids'),
            dtype=torch.long
        )
        all_input_mask = torch.tensor(
            self.select_field(features, 'input_mask'),
            dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            self.select_field(features, 'segment_ids'),
            dtype=torch.long
        )
        all_output_mask = torch.tensor(
            self.select_field(features, 'output_mask'),
            dtype=torch.bool
        )
        all_label = torch.tensor(
            [f.label for f in features],
            dtype=torch.long
        )
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    def select_field(self, features, field):
        return [[choice[field] for choice in feature.choices_features]
                for feature in features]

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
        """
        Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to
                the CLS token (0 for BERT, 2 for XLNet)
        """

        class InputFeatures(object):
            def __init__(self,  choices_features, label):

                self.choices_features = [
                    {
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'segment_ids': segment_ids,
                        'output_mask': output_mask,
                    }
                    for _, input_ids, input_mask, segment_ids,
                        output_mask in choices_features
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

            label = label_map[labels]
            choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
        features.append(InputFeatures(
                choices_features=choices_features,
                label=label
            ))

        return features

    def load_sparse_adj_data_with_contextnode(
            self,
            adj_concept_pairs,
            max_node_num,
            num_choice
    ):
        # this is actually n_questions x n_choices
        n_samples = len(adj_concept_pairs)
        # print("adj:", adj_concept_pairs)
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)  # default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)

        adj_lengths_ori = adj_lengths.clone()
        for idx, _data in tqdm(enumerate(adj_concept_pairs),
                               total=n_samples, desc='loading adj matrices'):
            adj, concepts, qm, am, cid2score = _data['adj'], \
                                               _data['concepts'],\
                                               _data['qmask'],\
                                               _data['amask'],\
                                               _data['cid2score']
            # adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type
            # '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            # concepts: np.array(num_nodes, ), where entry is concept id
            # qm: np.array(num_nodes, ), where entry is True/False
            # am: np.array(num_nodes, ), where entry is True/False
            # concepts = np.array(list(set(concepts)))
            # TODO: should be removed after fixing the api concept2id_api ()
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
            # this is the final number of nodes including
            # contextnode but excluding PAD
            num_concept = min(len(concepts), max_node_num - 1) + 1
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            # Prepare nodes
            concepts = concepts[:num_concept - 1]
            # To accomodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 1:num_concept] = torch.tensor(
                concepts + 1)
            # this is the "concept_id" for contextnode
            concept_ids[idx, 0] = 0

            # Prepare node scores
            if (cid2score is not None):
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1
                    assert _cid in cid2score
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            # Prepare node types
            node_type_ids[idx, 0] = 3  # context node
            node_type_ids[idx, 1:num_concept][torch.tensor(
                qm, dtype=torch.bool)[:num_concept - 1]] = 0
            node_type_ids[idx, 1:num_concept][torch.tensor(
                am, dtype=torch.bool)[:num_concept - 1]] = 1

            # Load adj
            # (num_matrix_entries, ), where each entry is coordinate
            ij = torch.tensor(adj.row, dtype=torch.int64)
            # (num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = torch.div(ij, n_node, rounding_mode='floor'), ij % n_node

            # Prepare edges
            i += 2
            j += 1
            # **** increment coordinate by 1, rel_id by 2 ****
            k += 1
            extra_i, extra_j, extra_k = [], [], []
            for _coord, q_tf in enumerate(qm):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if q_tf:
                    extra_i.append(0)  # rel from context node to question concept
                    extra_j.append(0)  # context node coordinate
                    extra_k.append(_new_coord)  # question concept coordinate
            for _coord, a_tf in enumerate(am):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if a_tf:
                    extra_i.append(1)  # rel from context node to answer concept
                    extra_j.append(0)  # context node coordinate
                    extra_k.append(_new_coord)  # answer concept coordinate

            half_n_rel += 2  # should be 19 now
            if len(extra_i) > 0:
                i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                k = torch.cat([k, torch.tensor(extra_k)], dim=0)

            mask = (j < max_node_num) & (k < max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = torch.cat((i, i + half_n_rel), 0), \
                      torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
            edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
            edge_type.append(i)  # each entry is [E, ]

        # list of size (n_questions, n_choices), where each entry is tensor[2, E]
        # this operation corresponds to .view(n_questions, n_choices)
        edge_index = list(map(list, zip(*(iter(edge_index),) * num_choice)))
        # list of size (n_questions, n_choices), where each entry is tensor[E, ]
        edge_type = list(map(list, zip(*(iter(edge_type),) * num_choice)))

        concept_ids, node_type_ids, node_scores, adj_lengths = [
            x.view(-1, num_choice, *x.size()[1:])
            for x in (concept_ids, node_type_ids, node_scores, adj_lengths)
        ]
        return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type)

    def to_numpy(self, x):
        if type(x) is not np.ndarray:
            x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
        return x

    def _predict(self):
        start = time.time()
        statements, grounded, graphs = self._prepare_data()
        self.num_choice = len(statements['choices'])
        self.graphs = graphs
        self.grounded = grounded
        model_type = self.lm_model.config.model_type
        assert self.model_path is not None

        features = self._convert_examples_to_features(
            examples=statements,
            max_seq_length=128,
            tokenizer=self.tokenizer,
            cls_token_at_end=bool(model_type in ['xlnet']),
            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=bool(model_type in ['roberta', 'albert']),
            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
            sequence_b_segment_id=0 if model_type in ['roberta', 'albert'] else 1
        )
        *data_tensors, all_label = self.convert_features_to_tensors(features)
        # end = time.time()
        # print(f"convert to features takes {end - start}")

        *test_decoder_data, test_adj_data = self.load_sparse_adj_data_with_contextnode(
            graphs,
            max_node_num=MAX_NODE_NUM,
            num_choice=self.num_choice,
        )

        input_data = [*data_tensors,*test_decoder_data, *test_adj_data]

        # start = time.time()
        with torch.no_grad():
            # logits, attn = self.model(*input_data)
            logits, attn, concept_ids, node_type_ids, edge_index_orig, edge_type_orig = self.model(*input_data, detail=True)


        self.attn = attn #
        self.concept_ids = concept_ids
        self.node_type_ids = node_type_ids
        # print(attn)
        # print(logits)
        # print(self.to_numpy(logits))
        # print(self.to_numpy(attn))
        # print(attn.size())
        # print(torch.nn.functional.softmax(logits, dim=1))
        # predictions = logits.argmax(1)  # [bsize, ]
        predictions, task_outputs = {}, {}
        if logits.size()[-1] != 1:
            probabilities = torch.softmax(logits, dim=-1)
            predictions["logits"] = probabilities
            labels = torch.argmax(predictions["logits"], dim=-1)
            task_outputs["labels"] = labels.tolist()
            self.correct_idx = labels.tolist()[0]

        end = time.time()
        print(f"prediction takes {end - start}")

        print(predictions)
        print(task_outputs)

        # print(f"The prediction of current input is : {self.inputs[predictions.detach().cpu().numpy()[0]][1]}")
        # return self.inputs[predictions.detach().cpu().numpy()[0]][1]

    def _get_attn(self):

        info = dict()
        info['concept_ids'] =  self.concept_ids.squeeze()  # (5,200)
        info['node_type_ids'] = self.node_type_ids.squeeze()  # (5,200)
        #0: question node
        #1: ans node
        #2: extra/other nodes
        #3: qa context node
        attn_h1 = self.attn[:self.num_choice]
        attn_h2 = self.attn[self.num_choice:]
        # info['attn'] = (attn_h1 + attn_h2) / 2  # (5,200)
        info['attn'] = (attn_h1 + attn_h2)  # (5,200)
        # info['attn'] = attn_h2   # (5,200)
        ## TODO:
        ## check only one atten
        ## check the sum

        a_idx = [info['node_type_ids'] == 1]
        a_id = set((info['concept_ids'][a_idx]-1).tolist())

        o_idx = [info['node_type_ids'] == 2]
        o_id = set((info['concept_ids'][o_idx]-1).tolist())

        q_idx = [info['node_type_ids']==0]
        q_id =  set((info['concept_ids'][ q_idx]-1).tolist())

        # # direction Q->O
        # q_idx = [info['node_type_ids']==0]
        # q_id =  set((info['concept_ids'][ q_idx]-1).tolist())
        # # for i in q_id:
        # #     print(id2concept[i])
        #
        # o_idx = [info['node_type_ids'] == 2]
        # o_id = set((info['concept_ids'][o_idx]-1).tolist())
        # # for i in o_id:
        # #     print(id2concept[i])
        # o_id_connected_with_q = [ ]
        # qo_paris = [list(i) for i in product(q_id,o_id)]
        # for pair in qo_paris:
        #     if (cpnet_simple.has_edge(pair[0], pair[1]) or cpnet_simple.has_edge(pair[1], pair[0])):
        #         o_id_connected_with_q.append(pair[1])
        #         # print(f"Q->O {id2concept[pair[0]]} and {id2concept[pair[1]]} are connected")
        #
        # o_id_connected_with_q = set(o_id_connected_with_q)
        # o_id_connected_with_q_idx = []
        # print("Direction Q->O:",[id2concept[i] for i in o_id_connected_with_q])
        #
        # #direction A->O
        # o_id_connected_with_a = []
        # a_idx = [info['node_type_ids'] == 1]
        # a_id = set((info['concept_ids'][a_idx]-1).tolist())
        # ao_paris = [list(i) for i in product(a_id, o_id)]
        # for pair in ao_paris:
        #     if (cpnet.has_edge(pair[0], pair[1]) or cpnet.has_edge(pair[1], pair[0])):
        #         o_id_connected_with_a.append(pair[1])
        #         # print(f"A->O {id2concept[pair[0]]} and {id2concept[pair[1]]} are connected")
        #
        # o_id_connected_with_a = set(o_id_connected_with_a)
        # o_id_connected_with_a_idx = []
        # print("Direction A->O:",[id2concept[i] for i in o_id_connected_with_a])
        #
        # # direction O->A
        # a_id_connected_with_o = []
        #
        # oa_paris = [list(i) for i in product(o_id, a_id)]
        # for pair in oa_paris:
        #     if (cpnet.has_edge(pair[0], pair[1]) or cpnet.has_edge(pair[1], pair[0])):
        #         a_id_connected_with_o.append(pair[1])
        #         # print(f"O->A {id2concept[pair[0]]} and {id2concept[pair[1]]} are connected")
        #
        # a_id_connected_with_o = set(a_id_connected_with_o)
        # a_id_connected_with_o_idx = []
        # a_id_connected_with_o_attn = []
        # for i in a_id:
        #     row_idx = []
        #     row_attn = []
        #     for n in range(info['attn'].shape[0]):
        #         idx = (info['concept_ids'][n]==(i+1)).nonzero()
        #
        #         try:
        #             row_idx.append(idx)
        #             row_attn.append(info['attn'][n][idx])
        #
        #         except IndexError:
        #             continue
        #     a_id_connected_with_o_idx.append(row_idx)
        #     a_id_connected_with_o_attn.append(row_attn)
        # re = dict()
        # for a,attn in zip(a_id,a_id_connected_with_o_attn):
        #     tmp_dict=dict()
        #     for i in range(len(attn)):
        #         tmp_dict['choice_'+str(i)] = attn[i]
        #     re[a] = tmp_dict
        # print("Direction O->A:",[id2concept[i] for i in a_id_connected_with_o])

        def bfs_attn(source, target):
            target_id_connected_with_source = []

            paris = [list(i) for i in product(source, target)]
            for pair in paris:
                if (cpnet.has_edge(pair[0], pair[1]) or cpnet.has_edge(pair[1], pair[0])):
                    target_id_connected_with_source.append(pair[1])

            target_id_connected_with_source = set(target_id_connected_with_source)
            target_id_connected_with_source_idx = []
            target_id_connected_with_source_attn = []
            for i in target:
                row_idx = []
                row_attn = []
                # for n in range(info['attn'].shape[0]):
                #     idx = (info['concept_ids'][n] == (i + 1)).nonzero()
                #
                #     try:
                #         row_idx.append(idx)
                #         row_attn.append(info['attn'][n][idx])
                #
                #     except IndexError:
                #         continue

                idx = (info['concept_ids'][self.correct_idx] == (i + 1)).nonzero()

                if idx.tolist()!=[]:

                    try:
                        row_idx.append(idx)
                        row_attn.append(info['attn'][self.correct_idx][idx])

                    except IndexError:

                        continue
                    target_id_connected_with_source_idx.append(row_idx)
                    target_id_connected_with_source_attn.append(row_attn)
            re = dict()
            for t, attn in zip(target, target_id_connected_with_source_attn):
                # tmp_dict = dict()
                # for i in range(len(attn)):
                #     tmp_dict['choice_' + str(i)] = attn[i]
                re[id2concept[t]] = float(attn[0][0][0])
            print("targer entities:", [id2concept[i] for i in target_id_connected_with_source])
            if 'ab_extra' in re:
                re.pop('ab_extra')

            re = dict(sorted(re.items(),key=operator.itemgetter(1),reverse=True ))
            # re = sorted(re.items(), key = lambda kv:(kv[1], kv[0]))
            # re_dict=dict()
            # for i in re:
            #     re_dict[i[0]]=i[1]
            return re





        qo_path = bfs_attn(q_id, o_id)
        ao_path = bfs_attn(a_id, o_id)
        oa_path = bfs_attn(o_id, a_id)

        print(qo_path)
        print(ao_path)
        print(oa_path)
        return (qo_path,ao_path,oa_path)


    def _get_info(self):


        def node_info(node_id: int, score_map: dict, grounded: dict):
            node = dict()
            node['id'] = node_id
            # node['name'] = kgapi.id2concept_api(np.abs(node_id))
            node['name'] = id2concept[np.abs(node_id)]
            node['description'] = ""
            node['q_node'] = False
            node['ans_node'] = False
            if node['name'] in grounded['qc']:
                node['q_node'] = True
            elif node['name'] in grounded['ac']:
                node['ans_node'] = True
            node['width'] = score_map[node_id]
            return node


        def edge_info(node_ids: list):
            id2relation = [
                'antonym',
                'atlocation',
                'capableof',
                'causes',
                'createdby',
                'isa',
                'desires',
                'hassubevent',
                'partof',
                'hascontext',
                'hasproperty',
                'madeof',
                'notcapableof',
                'notdesires',
                'receivesaction',
                'relatedto',
                'usedfor',
                    ]
            pair_list = [list(i) for i in list(product(node_ids, node_ids))]
            # edges = [cpnet[pair[0]][pair[1]][0] for pair in pair_list]
            edges = []

            for pair in pair_list:
                if cpnet_simple.has_edge(pair[0],pair[1]):
                    edges.append(cpnet[pair[0]][pair[1]][0])


            re = dict()

            for i, (node_pair,edge) in enumerate(zip(pair_list,edges)) :
                tmp_dict = dict()
                tmp_dict['source'] = node_pair[0]
                tmp_dict['target'] = node_pair[1]
                tmp_dict['weight'] = edge['weight']
                if edge['rel'] >= len(id2relation):
                    tmp_dict['label'] =id2relation[edge['rel']-len(id2relation)]
                else:
                    tmp_dict['label'] = id2relation[edge['rel']]
                re[i] = tmp_dict
            return re

        info = dict()
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
                info['nodes']['statement_' + str(i)][str(k)] = node_info(k, score, self.grounded[i])

            info['scores']['statement_' + str(i)] = score

            info['edges']['statement_' + str(i)] = dict()
            info['edges']['statement_' + str(i)] = edge_info(keys)

        info = json.dumps(info, indent=4)
        with open('info_demo.json', 'w') as f:
            f.write(info)

        return info



if __name__ == '__main__':
    model_path = DATA_DIR+"csqa_model_hf3.4.0.pt"
    # question = "The sanctions against the school were a punishing blow, and they seemed " \
    #            "to what the efforts the school had made to change?"
    # choices =  ["ignore", "enforce", "authoritarian", "yell at", "avoid"]
    # question = "There is a star at the center of what group of celestial bodies?"
    # choices = ["hollywood", "skyline", "outer space", "constellation", "solar system"]

    # question = "Google Maps and other highway and street GPS services have replaced what?"
    # choices = ["united states", "mexico", "countryside", "atlas", "oceans"]


    # question = "Crabs live in what sort of enviroment?"
    # choices = ["saltwater","galapagos","fish market"]

    question = "Where would you find a basement that can be accesssed with an elevator?"
    choices = ["closet","church","office building"]
    input = [[question, choice] for choice in choices]
    # input = {"question": question, "choices": choices}
    start = time.time()
    inf = Inference(inputs=input, model_path=model_path)
    inf._predict()
    inf._get_attn()
    end = time.time()
    print("Inference time: ", end-start)
