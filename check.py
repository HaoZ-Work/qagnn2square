import torch
# from graph_transformers.preprocess.kgapi import *
def check_info(num:int, attn:torch.tensor,node_ids:torch.tensor,scores:torch.tensor):
    '''
    Check the mapping between nodes and their score,attn

    Args:
        num: how many node to show
        attn:
        node_ids:
        scores:

    Return:
        None
    '''

    with open('data/concept.txt', "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    for i in range(attn.shape[0]):
        topk_attn,topk_idx = torch.topk(attn[i],num)

        topk_node_ids = node_ids[i][topk_idx].tolist()
        topk_score = scores[i][topk_idx]
        #print(topk_node_ids)
        print([id2concept[i] for i in  topk_node_ids])
        print(topk_attn.tolist())
        print(topk_score.tolist())
        print("*"*20)











def main():
    attn = torch.load('attn.pt')
    node_ids = torch.load('node_ids.pt')
    scores = torch.load('scores.pt')

    check_info(10,attn,node_ids,scores)

if __name__ == '__main__':
    main()