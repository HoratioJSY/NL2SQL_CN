import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import collections
import torch.nn as nn
import numpy as np
from pytorch_transformers import BertTokenizer, BertModel


class BertEmbedding(nn.Module):
    def __init__(self, N_word, gpu, our_model, bert_path):
        super(BertEmbedding, self).__init__()
        self.N_word = N_word
        self.our_model = our_model
        self.gpu = gpu

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.Token2ID = self.load_vocab(bert_path + 'vocab.txt')
        self.bert_model = BertModel.from_pretrained(bert_path)

        self.bert_model.eval()
        if self.gpu:
            self.bert_model.to('cuda')
            print('Using BERT in CUDA')

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        Toke2ID = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                Toke2ID[token] = index
                index += 1
        return Toke2ID

    def gen_x_batch(self, q):
        B = len(q)
        tokenizered_q = []
        val_len = np.zeros(B, dtype=np.int64)

        # 一条Query及对应的表头，表头为列名
        for i, one_q in enumerate(q):
            contain_token = []
            for tok in one_q:
                if self.Token2ID.__contains__(tok):
                    contain_token.append(tok)
                else:
                    contain_token.append('[UNK]')

            one_token = ['[CLS]'] + contain_token + ['[SEP]']
            assert len(one_q)+2 == len(one_token)

            tokenizered_q.append(one_token)
            val_len[i] = len(one_token)
        max_len = max(val_len)
        assert len(tokenizered_q) == B

        tokens_id_tensor = torch.zeros([B, max_len], dtype=torch.int64)
        # tokens if used in attention, The mask has 1 for real tokens and 0 for padding tokens
        attention_mask = torch.zeros([B, max_len], dtype=torch.long)

        for i, sentence in enumerate(tokenizered_q):
            tokens_id_tensor[i, :val_len[i]] = torch.tensor([self.tokenizer.convert_tokens_to_ids(sentence)])
            attention_mask[i, :val_len[i]] = torch.ones([1, val_len[i]], dtype=torch.long)

        if self.gpu:
            tokens_id_tensor = tokens_id_tensor.to('cuda')
            attention_mask = attention_mask.to('cuda')

        with torch.no_grad():
            hidden_state, _ = self.bert_model(tokens_id_tensor, attention_mask=attention_mask)
        # print('hidden state size1:', hidden_state.shape)
        return hidden_state, val_len

    def gen_col_batch(self, cols):
        # cols = [[[tok_1, tok_2], column_2, ...], table_2, ...]
        # col_len is the column num for different table
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

        for i, one_str in enumerate(str_list):

            contain_token = []
            for tok in one_str:
                if self.Token2ID.__contains__(tok):
                    contain_token.append(tok)
                else:
                    contain_token.append('[UNK]')
            tokenizered_chr.append(['[CLS]'] + contain_token + ['[SEP]'])
            val_len[i] = len(tokenizered_chr[-1])
        max_len = max(val_len)
        assert len(tokenizered_chr) == B

        tokens_id_tensor = torch.zeros([B, max_len], dtype=torch.int64)
        for i, sentence in enumerate(tokenizered_chr):
            tokens_id_tensor[i, :val_len[i]] = torch.tensor([self.tokenizer.convert_tokens_to_ids(sentence)])

        attention_mask = torch.zeros([B, max_len], dtype=torch.long)
        for i, sample_length in enumerate(val_len):
            attention_mask[i, :sample_length] = torch.ones([1, sample_length], dtype=torch.long)

        if self.gpu:
            tokens_id_tensor = tokens_id_tensor.to('cuda')
            attention_mask = attention_mask.to('cuda')

        with torch.no_grad():
            hidden_state, _ = self.bert_model(tokens_id_tensor, attention_mask=attention_mask)
        # print('hidden state size2:', hidden_state.shape)

        return hidden_state, val_len

    def gen_table_batch(self, table_content):
        # table_content = [[[str1, str2, ...], column2, ...], table2, ...]
        tokenizered_chr = []
        table_config = []

        for i, table in enumerate(table_content):
            value_num = []
            for j, column_content in enumerate(table):
                # the num of column in total is the length of tokenizered_chr
                value_num.append(len(column_content))
                contain_token = ['[CLS]']
                for value in column_content:
                    for token in str(value):
                        if self.Token2ID.__contains__(token):
                            contain_token.append(token)
                        else:
                            contain_token.append('[UNK]')
                    contain_token = contain_token + ['[SEP]']
                tokenizered_chr.append(contain_token)
            # a list for columns that contain all value num, [[value num in column 1, ...], table2, ...]
            table_config.append(value_num)

        max_token_len = max([len(x) for x in tokenizered_chr])
        tokens_id_tensor = torch.zeros([len(tokenizered_chr), max_token_len], dtype=torch.int64)
        attention_mask = torch.zeros([len(tokenizered_chr), max_token_len], dtype=torch.long)
        for i, column_token in enumerate(tokenizered_chr):
            tokens_id_tensor[i, :len(column_token)] = torch.tensor([self.tokenizer.convert_tokens_to_ids(column_token)])
            attention_mask[i, :len(column_token)] = torch.ones([1, len(column_token)], dtype=torch.long)

        if self.gpu:
            tokens_id_tensor = tokens_id_tensor.to('cuda')
            attention_mask = attention_mask.to('cuda')
        with torch.no_grad():
            hidden_state, _ = self.bert_model(tokens_id_tensor, attention_mask=attention_mask)
        assert hidden_state.size() == (len(tokenizered_chr), max_token_len, self.N_word)

        # average every token's embedding as value representation
        max_value_num = max([max(column_config) for column_config in table_config])
        embed_tensor = torch.zeros([len(tokenizered_chr), max_value_num, self.N_word])

        for i, table in enumerate(table_config):
            for j, value_num in enumerate(table):
                assert tokenizered_chr[i*len(table)+j].count('[SEP]') == value_num

                value_index = [0]
                for _ in range(value_num):
                    value_index.append(tokenizered_chr[i*len(table)+j].index('[SEP]', value_index[-1]+1))

                # caculate average
                for k in range(value_num):
                    embed_tensor[i+j, k, :] = \
                        torch.mean(hidden_state[i+j, value_index[k]+1:value_index[k+1]+1, :], dim=0, keepdim=True)
        return embed_tensor, table_config

    def condition_value_batch_backpack(self, value_list, max_value_length):
        # value_list=[[value list for condition one],...]
        tokenizered_chr = []

        # tokenizered_chr=[['[CLS]', 'tok_1', 'tok_2', '[SEP]', 'tok_3', ...,'[SEP]'],...]
        for one_condition_value in value_list:
            contain_token = ['[CLS]']
            for value in one_condition_value:
                for chr in value:
                    if self.Token2ID.__contains__(chr):
                        contain_token.append(chr)
                    else:
                        contain_token.append('[UNK]')
                contain_token = contain_token + ['[SEP]']
            tokenizered_chr.append(contain_token)

        max_token_len = max([len(x) for x in tokenizered_chr])
        tokens_id_tensor = torch.zeros([len(tokenizered_chr), max_token_len], dtype=torch.int64)
        attention_mask = torch.zeros([len(tokenizered_chr), max_token_len], dtype=torch.long)

        for i, cond_column in enumerate(tokenizered_chr):
            tokens_id_tensor[i, :len(cond_column)] = torch.tensor([self.tokenizer.convert_tokens_to_ids(cond_column)])
            attention_mask[i, :len(cond_column)] = torch.ones([1, len(cond_column)], dtype=torch.long)

        if self.gpu:
            tokens_id_tensor = tokens_id_tensor.to('cuda')
            attention_mask = attention_mask.to('cuda')
        with torch.no_grad():
            hidden_state, _ = self.bert_model(tokens_id_tensor, attention_mask=attention_mask)

        assert hidden_state.size() == (len(tokenizered_chr), max_token_len, self.N_word)

        # average every token's embedding as value representation
        embed_tensor = torch.zeros([len(tokenizered_chr), max_value_length, self.N_word])

        for j, one_cond_list in enumerate(value_list):
            assert tokenizered_chr[j].count('[SEP]') == len(one_cond_list)
            value_index = [0]
            for _ in range(len(one_cond_list)):
                value_index.append(tokenizered_chr[j].index('[SEP]', value_index[-1] + 1))

            # caculate average
            for k in range(len(one_cond_list)):
                embed_tensor[j, k, :] = \
                    torch.mean(hidden_state[j, value_index[k] + 1:value_index[k + 1] + 1, :], dim=0, keepdim=True)
        return embed_tensor

    def condition_value_batch(self, value_list, max_value_length):
        # value_list=[[value list for condition one],...]
        tokenizered_chr = []
        value_num = [len(x) for x in value_list]

        # tokenizered_chr=[['[CLS]', 'tok_1', 'tok_2', '[SEP]'],...]
        for one_condition_value in value_list:
            for value in one_condition_value:
                contain_token = []
                for chr in value:
                    if self.Token2ID.__contains__(chr):
                        contain_token.append(chr)
                    else:
                        contain_token.append('[UNK]')
                tokenizered_chr.append(['[CLS]'] + contain_token + ['[SEP]'])

        assert len(tokenizered_chr) == np.array(value_num).sum()

        max_token_len = max([len(x) for x in tokenizered_chr])
        tokens_id_tensor = torch.zeros([len(tokenizered_chr), max_token_len], dtype=torch.int64)
        attention_mask = torch.zeros([len(tokenizered_chr), max_token_len], dtype=torch.long)

        # print('tokens_id_tensor', tokens_id_tensor.shape)

        for i, value_seq in enumerate(tokenizered_chr):
            tokens_id_tensor[i, :len(value_seq)] = torch.tensor([self.tokenizer.convert_tokens_to_ids(value_seq)])
            attention_mask[i, :len(value_seq)] = torch.ones([1, len(value_seq)], dtype=torch.long)

        if self.gpu:
            tokens_id_tensor = tokens_id_tensor.to('cuda')
            attention_mask = attention_mask.to('cuda')
        with torch.no_grad():
            hidden_state = []
            b_bert = 64
            for b in range(len(tokenizered_chr)//b_bert):
                h, _ = self.bert_model(tokens_id_tensor[b*b_bert:b*b_bert+b_bert, :],
                                       attention_mask=attention_mask[b*b_bert:b*b_bert+b_bert, :])
                hidden_state.append(h)
            h, _ = self.bert_model(tokens_id_tensor[(len(tokenizered_chr)//b_bert)*b_bert:, :],
                                   attention_mask=attention_mask[(len(tokenizered_chr)//b_bert)*b_bert:, :])
            hidden_state.append(h)
            hidden_state = torch.cat(hidden_state, dim=0)
        assert hidden_state.size() == (len(tokenizered_chr), max_token_len, self.N_word)

        # average every token's embedding as value representation
        embed_tensor = torch.zeros([len(value_list), max_value_length, self.N_word])

        value_num_mark = 0
        for j, one_cond in enumerate(value_num):
            for k in range(one_cond):
                embed_tensor[j, k, :] = torch.mean(hidden_state[value_num_mark, 1:-1, :], dim=0, keepdim=True)
                value_num_mark += 1
        return embed_tensor