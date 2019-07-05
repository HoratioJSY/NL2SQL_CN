import torch
import torch.nn as nn
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class BertEmbedding(nn.Module):
    def __init__(self, N_word, gpu, SQL_TOK, our_model):
        super(BertEmbedding, self).__init__()
        self.N_word = N_word
        self.our_model = our_model
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        # self.tokenizer = BertTokenizer.from_pretrained('/Users/horatio_jsy/pylib/NL2SQL_CN/pre_trained/')
        # self.bert_model = BertModel.from_pretrained('/Users/horatio_jsy/pylib/NL2SQL_CN/pre_trained/')
        self.tokenizer = BertTokenizer.from_pretrained('/content/drive/My Drive/pre_trained/')
        self.bert_model = BertModel.from_pretrained('/content/drive/My Drive/pre_trained/')
        self.bert_model.eval()

    def gen_x_batch(self, q, col):
        B = len(q)
        tokenizered_q = []
        tokens_id_list = []
        val_len = np.zeros(B, dtype=np.int64)

        # 一条Query及对应的表头，表头为列名
        for i, one_q in enumerate(q):
            # print('one_q', one_q)
            # one_token = ['[CLS]'] + self.tokenizer.tokenize(''.join(one_q)) + ['[SEP]']
            one_token = ['[CLS]'] + one_q + ['[SEP]']
            # print('one_token', one_token)
            assert len(one_q)+2 == len(one_token)
            tokenizered_q.append(one_token)
            tokens_id_list.append(self.tokenizer.convert_tokens_to_ids(one_token))
            val_len[i] = len(one_token)
        max_len = max(val_len)
        # print(val_len)
        # print('max len:', max_len)

        tokens_id_array = np.zeros([B, max_len], dtype=np.int64)
        for i in range(B):
            for t in range(len(tokens_id_list[i])):
                tokens_id_array[i, t] = tokens_id_list[i][t]
        tokens_id_tensor = torch.from_numpy(tokens_id_array)
        segments_tensor = torch.zeros([B, max_len], dtype=torch.int64)

        if self.gpu:
            tokens_id_tensor = tokens_id_tensor.cuda()
            segments_tensor = segments_tensor.cuda()

        with torch.no_grad():
            hidden_state, _ = self.bert_model(tokens_id_tensor, segments_tensor, output_all_encoded_layers=False)
        # print('hidden state size1:', hidden_state.shape)
        return hidden_state, val_len

    def gen_col_batch(self, cols):
        # 一个Query有一张表，一张表有多个特征列名，每个特征列名又能Tokenizer
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)

        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        B = len(str_list)

        val_len = np.zeros(B, dtype=np.int64)
        tokenizered_chr = []
        tokens_id = []

        for i, one_str in enumerate(str_list):
            tokenizered_chr.append(['[CLS]'] + one_str + ['[SEP]'])
            tokens_id.append(self.tokenizer.convert_tokens_to_ids(tokenizered_chr[-1]))
            val_len[i] = len(tokenizered_chr[-1])
        max_len = max(val_len)

        tokens_id_array = np.zeros((B, max_len), dtype=np.int64)
        for i in range(B):
            for t in range(len(tokens_id[i])):
                tokens_id_array[i, t] = tokens_id[i][t]
        tokens_id_tensor = torch.from_numpy(tokens_id_array)
        segments_tensor = torch.zeros([B, max_len], dtype=torch.int64)

        if self.gpu:
            tokens_id_tensor = tokens_id_tensor.cuda()
            segments_tensor = segments_tensor.cuda()

        with torch.no_grad():
            hidden_state, _ = self.bert_model(tokens_id_tensor, segments_tensor, output_all_encoded_layers=False)
        # print('hidden state size2:', hidden_state.shape)

        return hidden_state, val_len