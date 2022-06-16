import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.conceptnet import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM

input_paths = {
    # 'csqa': {
    #     'train': './data/csqa/train_rand_split.jsonl',
    #     'dev': './data/csqa/dev_rand_split.jsonl',
    #     'test': './data/csqa/test_rand_split_no_answers.jsonl',
    # },

    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'mytest': {
        'test': './data/mytest/mytest_data.jsonl',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },

    'mytest': {
        'statement': {
            'test': './data/mytest/statement/test.statement.jsonl',
        },
        'grounded': {

            'test': './data/mytest/grounded/test.grounded.jsonl',
        },
        'graph': {

            'adj-test': './data/mytest/graph/test.graph.adj.pk',
        },
    },





}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common'], choices=['common', 'csqa', 'hswag', 'anli', 'exp', 'scitail', 'phys', 'socialiqa', 'obqa', 'obqa-fact', 'make_word_vocab','mytest'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],

        'mytest': [
            {'func': convert_to_entailment,
             'args': (input_paths['mytest']['test'], output_paths['mytest']['statement']['test'])},
            {'func': ground, 'args': (output_paths['mytest']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['mytest']['grounded']['test'],
                                      args.nprocs)},

            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (
            output_paths['mytest']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
            output_paths['cpnet']['vocab'], output_paths['mytest']['graph']['adj-test'], args.nprocs)},
        ],


    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
