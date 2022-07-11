import time
import numpy as np
from graph_transformers.modelling import (
    roberta,
    qagnn
)
from graph_transformers.preprocess import (
    statement,
    grounding,
    graph,
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

            def __init__(self, input_ids, input_mask, segment_ids, output_mask, label):

                self.input_ids = input_ids,
                self.input_mask = input_mask,
                self.segment_ids = segment_ids,
                self.output_mask = output_mask,
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
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.long)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long)
            output_mask = torch.tensor(output_mask, dtype=torch.bool)
            label = torch.tensor([label], dtype=torch.long)

            # choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
            features.append(InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                output_mask=output_mask,
                label=label
            ))

        return features

    def _predict(self):

        statements, grounded, graphs = self._prepare_data()
        model_type = "roberta"
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

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
        print("after loading")

        # load the model
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
        print(features)
        *input_data, labels = features

        logits, _, = model(*input_data)
        predictions = logits.argmax(1)  # [bsize, ]
        print(f"The prediction of current input is : {predictions}")


if __name__ == '__main__':
    model_path = "data/csqa_model_hf3.4.0.pt"
    question = "There is a star at the center of what group of celestial bodies?"
    choices = ["hollywood", "skyline", "outer space", "constellation", "solar system"]
    input = {"question": question, "choices": choices}
    start = time.time()
    inf = Inference(inputs=input, use_lm=True, model_path=model_path)
    inf._predict()
    end = time.time()
    print("Inference time: ", end-start)
