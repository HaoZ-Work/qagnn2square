'''
A simple inference pipeline with csqs format data

'''

from qagnn_utils.convert_csqa import convert_to_entailment
from qagnn_utils.grounding import ground
from qagnn_utils.graph import generate_adj_data_from_grounded_concepts__use_LM_api
from qagnn_utils.modeling_qagnn import *
from qagnn_utils.parser_utils import *

from qagnn_utils.data_utils import *

import torch
import json
import datetime
import pprint

DECODER_DEFAULT_LR = {
    'mytest': 1e-3,
    'csqa': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

class LM_QAGNN_DataLoader_inference(object):

    def __init__(self, args,test_statement_json,test_adj,

                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,

                 subsample=1.0):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device



        model_type = MODEL_NAME_TO_CLASS[model_name]
        # print ('train_statement_path', train_statement_path)
        # self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        # self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)
        #
        # num_choice = self.train_encoder_data[0].size(1)
        # csqs has 5 choices for Q
        num_choice =5
        self.num_choice = num_choice
        # print ('num_choice', num_choice)
        # *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, num_choice, args)
        #
        # *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, num_choice, args)
        # assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        # assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        # if test_statement_json is not None:
        #     # seems like get tensor from LM model
        #     self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_json, model_type, model_name, max_seq_length)
        #
        #
        #
        #     # the return is :  concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type)
        #     *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj, max_node_num, num_choice, args)
        #     assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)
        #     print("check point ")

        self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_json, model_type,
                                                                                       model_name, max_seq_length)

        *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj, max_node_num,
                                                                                            num_choice, args)
        assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in
                   [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        # if self.is_inhouse:
        #     with open(inhouse_train_qids_path, 'r') as fin:
        #         inhouse_qids = set(line.strip() for line in fin)
        #     self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
        #     self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        # if subsample < 1.:
        #     n_train = int(self.train_size() * subsample)
        #     assert n_train > 0
        #     if self.is_inhouse:
        #         self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
        #     else:
        #         self.train_qids = self.train_qids[:n_train]
        #         self.train_labels = self.train_labels[:n_train]
        #         self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
        #         self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
        #         self.train_adj_data = self.train_adj_data[:n_train]
        #         assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        #     assert self.train_size() == n_train

    # def train_size(self):
    #     return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)
    #
    # def dev_size(self):
    #     return len(self.dev_qids)

    # def test_size(self):
    #     # if self.is_inhouse:
    #     #     return self.inhouse_test_indexes.size(0)
    #     # else:
    #     return len(self.test_qids) if hasattr(self, 'test_qids') else 0
    #
    # def train(self):
    #     if self.is_inhouse:
    #         n_train = self.inhouse_train_indexes.size(0)
    #         train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
    #     else:
    #         train_indexes = torch.randperm(len(self.train_qids))
    #     return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train', self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
    #
    # def train_eval(self):
    #     return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
    #
    # def dev(self):
    #     return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)
        print(self.args)
    def test(self):
        # if self.is_inhouse:
        #     return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        # else:
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data)



        # return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size,
        #                                            torch.arange(len(self.test_qids)), self.test_qids, self.test_labels,
        #                                            tensors0=self.test_encoder_data, tensors1=self.test_decoder_data,
        #                                            adj_data=self.test_adj_data)


def eval_detail(args,test_statement_json,test_graph):
    assert args.load_model_path is not None
    model_path = args.load_model_path


    ## only for medqa data
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    # print('| num_concepts: {} |'.format(concept_num))


    # load the model
    model_state_dict, old_args = torch.load(model_path, map_location=torch.device('cpu'))


    # create the model template
    model = LM_QAGNN(old_args, old_args.encoder, k=old_args.k, n_ntype=4, n_etype=old_args.num_relation, n_concept=concept_num,
                               concept_dim=old_args.gnn_dim,
                               concept_in_dim=concept_dim,
                               n_attention_head=old_args.att_head_num, fc_dim=old_args.fc_dim, n_fc_layer=old_args.fc_layer_num,
                               p_emb=old_args.dropouti, p_gnn=old_args.dropoutg, p_fc=old_args.dropoutf,
                               pretrained_concept_emb=cp_emb, freeze_ent_emb=old_args.freeze_ent_emb,
                               init_range=old_args.init_range,
                               encoder_config={})

    #load the model
    model.load_state_dict(model_state_dict)


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

    # statement_dic = {}
    # for statement_path in (args.train_statements, args.dev_statements, args.test_statements):
    #     statement_dic.update(load_statement_dict(statement_path))

    use_contextualized = 'lm' in old_args.ent_emb

    # print ('inhouse?', args.inhouse)
    #
    # print ('args.train_statements', args.train_statements)
    # print ('args.dev_statements', args.dev_statements)
    # print ('args.test_statements', args.test_statements)
    # print ('args.train_adj', args.train_adj)
    # print ('args.dev_adj', args.dev_adj)
    # print ('args.test_adj', args.test_adj)
    print(old_args.encoder)


    start = time.time()
    dataset = LM_QAGNN_DataLoader_inference(args,test_statement_json,test_graph,

                                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                           device=(device0, device1),
                                           model_name=old_args.encoder,
                                           max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                                           subsample=args.subsample)

    # save_test_preds = args.save_model
    # dev_acc = evaluate_accuracy(dataset.dev(), model)
    # print('dev_acc {:7.4f}'.format(dev_acc))
    # if not save_test_preds:
    #     test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
    # else:
    eval_set = dataset.test()

    end = time.time()
    print(f"dataloader takes {end-start}")


        #print(next(iter(eval_set)))
        #
    # dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        # preds_path = os.path.join(args.save_dir, 'test_preds_{}.csv'.format(dt))
        # with open(preds_path, 'w') as f_preds:
        #     with torch.no_grad():
        #         for qids, labels, *input_data in tqdm(eval_set):
        #
        #             logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data, detail=True)
        #             predictions = logits.argmax(1) #[bsize, ]
        #             print(f"The prediction of current input is : {predictions}")


    for qids, labels, *input_data in eval_set:

        logits, _,= model(*input_data)
        predictions = logits.argmax(1)  # [bsize, ]
        print(f"The prediction of current input is : {predictions}")


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/qagnn/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model_path', default=None)

    # data
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True,
                        help='use cached data to accelerate data loading')

    # model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True,
                        help='freeze entity embedding layer')

    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')

    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float,
                        help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--fp16', default=False, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')


    mytest_data_json= {"id": "000990552527b1353f98f1e1a7dfc643", "question": {"question_concept": "star", "choices": [{"label": "A", "text": "hollywood"}, {"label": "B", "text": "skyline"}, {"label": "C", "text": "outer space"}, {"label": "D", "text": "constellation"}, {"label": "E", "text": "solar system"}], "stem": "There is a star at the center of what group of celestial bodies?"}}



    ## get the statement for one test recording
    test_statment = convert_to_entailment(mytest_data_json)
    #{'id': '000990552527b1353f98f1e1a7dfc643', 'question': {'question_concept': 'star', 'choices': [{'label': 'A', 'text': 'hollywood'}, {'label': 'B', 'text': 'skyline'}, {'label': 'C', 'text': 'outer space'}, {'label': 'D', 'text': 'constellation'}, {'label': 'E', 'text': 'solar system'}], 'stem': 'There is a star at the center of what group of celestial bodies?'}, 'statements': [{'label': True, 'statement': 'There is a star at the center of hollywood group of celestial bodies.'}, {'label': False, 'statement': 'There is a star at the center of skyline group of celestial bodies.'}, {'label': False, 'statement': 'There is a star at the center of outer space group of celestial bodies.'}, {'label': False, 'statement': 'There is a star at the center of constellation group of celestial bodies.'}, {'label': False, 'statement': 'There is a star at the center of solar system group of celestial bodies.'}]}


    ##get the grounded
    start = time.time()
    test_grounded = ground(test_statment,cpnet_vocab_path='./data/cpnet/concept.txt',pattern_path='./data/cpnet/matcher_patterns.json' ,num_processes=1)
    end = time.time()
    print(f"grounded taked {end-start}")
    #
    # {'sent': 'There is a star at the center of hollywood group of celestial bodies.', 'ans': 'hollywood',
    #  'qc': ['bodies', 'body', 'celestial', 'celestial_bodies', 'celestial_body', 'center', 'group', 'star'],
    #  'ac': ['hollywood']}
    # {'sent': 'There is a star at the center of skyline group of celestial bodies.', 'ans': 'skyline',
    #  'qc': ['bodies', 'body', 'celestial', 'celestial_bodies', 'celestial_body', 'center', 'group', 'star'],
    #  'ac': ['skyline']}
    # {'sent': 'There is a star at the center of outer space group of celestial bodies.', 'ans': 'outer space',
    #  'qc': ['bodies', 'body', 'celestial', 'celestial_bodies', 'celestial_body', 'center', 'group', 'space_group',
    #         'star'], 'ac': ['outer', 'outer_space', 'space']}
    # {'sent': 'There is a star at the center of constellation group of celestial bodies.', 'ans': 'constellation',
    #  'qc': ['bodies', 'body', 'celestial', 'celestial_bodies', 'celestial_body', 'center', 'group', 'star'],
    #  'ac': ['constellation']}
    # {'sent': 'There is a star at the center of solar system group of celestial bodies.', 'ans': 'solar system',
    #  'qc': ['bodies', 'body', 'celestial', 'celestial_bodies', 'celestial_body', 'center', 'group', 'star'],
    #  'ac': ['solar', 'solar_system', 'system']}

    # get the graph
    test_graph_adj=generate_adj_data_from_grounded_concepts__use_LM_api(test_statment,test_grounded,cpnet_graph_path='./data/cpnet/conceptnet.en.pruned.graph',cpnet_vocab_path='./data/cpnet/concept.txt',num_processes=6)
    # test graph_adj:
    # adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
    # concepts ids
    # qmask that specifices whether a node is a question concept
    # amask that specifices whether a node is a answer concept
    # cid2score that maps a concept id to its relevance score given the QA context




    ### rewrite eval_detail()

    start = time.time()
    eval_detail(args,test_statement_json=test_statment,test_graph=test_graph_adj)
    end = time.time()

    print(f"QAGNN takes{end-start}")

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    print(f"The whole inference takes {end-start}")