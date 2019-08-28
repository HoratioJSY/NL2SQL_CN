import torch
import re
import torch.nn as nn
import numpy as np
from sqlnet.model.modules.net_utils import run_lstm, col_name_encode
from sqlnet.utils import cn_to_num

class SQLNetCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, gpu, embed_layer, use_table):
        super(SQLNetCondPredictor, self).__init__()
        self.N_h = N_h * 2
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.gpu = gpu
        self.N_depth = N_depth + 1
        self.emb_layer = embed_layer
        self.use_table = use_table

        # predict condition number
        self.cond_num_lstm = nn.LSTM(input_size=N_word, hidden_size=int(self.N_h/2), num_layers=self.N_depth,
                                     batch_first=True, dropout=0.3, bidirectional=True)
        self.cond_num_att = nn.Linear(self.N_h, 1)
        self.cond_num_out = nn.Sequential(nn.Linear(self.N_h, self.N_h),
                nn.Tanh(), nn.Linear(self.N_h, 5))

        # predict condition column
        self.cond_num_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(self.N_h/2), num_layers=self.N_depth,
                                         batch_first=True, dropout=0.3, bidirectional=True)
        self.cond_num_col_att = nn.Linear(self.N_h, 1)
        self.cond_num_col2hid1 = nn.Linear(self.N_h, self.N_depth*self.N_h)
        self.cond_num_col2hid2 = nn.Linear(self.N_h, self.N_depth*self.N_h)

        self.cond_col_lstm = nn.LSTM(input_size=N_word, hidden_size=int(self.N_h/2), num_layers=self.N_depth,
                                     batch_first=True, dropout=0.3, bidirectional=True)

        # Using column attention on where predicting
        self.cond_col_att = nn.Linear(self.N_h, self.N_h)

        self.cond_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(self.N_h/2), num_layers=self.N_depth,
                                         batch_first=True, dropout=0.3, bidirectional=True)
        self.cond_col_out_K = nn.Linear(self.N_h, self.N_h)
        self.cond_col_out_col = nn.Linear(self.N_h, self.N_h)
        self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(self.N_h, 1))

        # predict condition operator
        self.cond_op_lstm = nn.LSTM(input_size=N_word, hidden_size=int(self.N_h/2), num_layers=self.N_depth,
                                    batch_first=True, dropout=0.3, bidirectional=True)
        # Using column attention
        self.cond_op_att = nn.Linear(self.N_h, self.N_h)

        self.cond_op_out_K = nn.Linear(self.N_h, self.N_h)

        self.cond_op_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(self.N_h/2), num_layers=self.N_depth,
                                        batch_first=True, dropout=0.3, bidirectional=True)
        self.cond_op_out_col = nn.Linear(self.N_h, self.N_h)
        self.cond_op_out = nn.Sequential(nn.Linear(self.N_h, self.N_h), nn.Tanh(),
                nn.Linear(self.N_h, 4))

        # predict condition value
        self.cond_str_lstm = nn.LSTM(input_size=N_word, hidden_size=int(self.N_h/2), num_layers=self.N_depth,
                                     batch_first=True, dropout=0.3, bidirectional=True)
        self.cond_str_decoder = nn.LSTM(input_size=self.max_tok_num, hidden_size=self.N_h, num_layers=self.N_depth,
                                        batch_first=True, dropout=0.3)
        self.cond_str_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(self.N_h/2), num_layers=self.N_depth,
                                         batch_first=True, dropout=0.3, bidirectional=True)

        self.cond_str_out_g = nn.Linear(self.N_h, self.N_h)
        self.cond_str_out_h = nn.Linear(self.N_h, self.N_h)
        self.cond_str_out_col = nn.Linear(self.N_h, self.N_h)
        self.cond_str_out = nn.Sequential(nn.ReLU(), nn.Linear(self.N_h, 1))

        self.softmax = nn.Softmax(dim=-1)

    def gen_gt_batch(self, split_tok_seq):
        B = len(split_tok_seq)

        # The max table content seq len in the batch
        max_len = max([max([len(tok) for tok in tok_seq]+[0]) for
            tok_seq in split_tok_seq]) - 1
        if max_len < 1:
            max_len = 1

        # 4 is max num of condition clause
        ret_array = np.zeros((
            B, 4, max_len, self.max_tok_num), dtype=np.float32)
        ret_len = np.zeros((B, 4))

        for b, tok_seq in enumerate(split_tok_seq):
            idx = 0

            # tok_seq is a list of [index] in one query
            for idx, one_tok_seq in enumerate(tok_seq):
                out_one_tok_seq = one_tok_seq[:-1]
                # record condition's value length, for different table and column
                ret_len[b, idx] = len(out_one_tok_seq)
                for t, tok_id in enumerate(out_one_tok_seq):
                    ret_array[b, idx, t, tok_id] = 1

            # pad for condition num
            if idx < 3:
                ret_array[b, idx+1:, 0, 1] = 1
                ret_len[b, idx+1:] = 1

        # [batch_size, condition num, max value length, max_tok_num=200]
        ret_inp_var = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp_var = ret_inp_var.to('cuda')

        return ret_inp_var, ret_len

    def cond_value_batch(self, table_content, query_seq, cond_num, chosen_col, cond_op_score):
        B = len(query_seq)
        assert len(table_content) == B

        # operator score to prediction
        cond_op = []
        for b in range(B):
            cond_op.extend([cond_op_score[b, num] for num in range(cond_num[b])])
        cond_op = torch.stack(cond_op)

        # print('cond_op.shape', cond_op.shape)
        # print('cond_num', cond_num)
        cond_op = torch.argmax(cond_op, dim=-1)
        assert cond_op.shape[-1] == torch.sum(cond_num)

        cond_column = []
        for x in chosen_col:
            cond_column.extend(x)
        assert len(cond_column) == torch.sum(cond_num)

        pattern = re.compile(r'[两\-一二三四五六七八九十.百千万亿年\d]+')
        num_dict = {'十': '0', '百': '00', '千': "000", '万': '0000', '亿': '00000000'}
        cond_value = []

        cond_num_mark = 0
        # len(cond_seq)=query_number
        for i, one_query in enumerate(query_seq):
            selected_num = pattern.findall(''.join(one_query))
            for j, element in enumerate(selected_num):
                if re.search(r'[\u4e00-\u9fa5]', element) is not None:
                    selected_num[j] = cn_to_num(element, selected_num)

                zero_nums = []
                for e_str in element:
                    if num_dict.__contains__(e_str):
                        zero_nums.append(num_dict[e_str])
                        selected_num.append('1' + num_dict[e_str])
                        if len(zero_nums) >= 2:
                            selected_num.append('1' + ''.join(zero_nums))

            # selected_num are proposed by query
            selected_num.extend(re.findall(r'[0-9]+', ''.join(one_query)))
            selected_num = list(set(selected_num))
            selected_table = table_content[i]

            for one_condition in range(cond_num[i]):

                # selected_column are proposed by table content
                selected_column = selected_table[cond_column[cond_num_mark]]
                for e, element in enumerate(selected_column):
                    if re.search(".0$", element) is not None:
                        selected_column[e] = re.sub(r'.0$', '', element)

                # select proposal by op prediction
                if cond_op[cond_num_mark] >= 2:
                    one_value = selected_column
                else:
                    # print('selected_num', selected_num)
                    one_value = selected_num

                one_value.extend('UNK')
                cond_value.append(one_value)
                cond_num_mark += 1

        assert cond_num_mark == torch.sum(cond_num) == len(cond_value)
        max_value_length = max([len(x) for x in cond_value])
        return cond_value, max_value_length

    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len,
            col_len, table_content, q_seq, gt_where, gt_cond):
        max_x_len = max(x_len)

        # batch size， x_len is a list for token nums of all queries
        B = len(x_len)

        # Predict the number of conditions
        # use column embeddings to calculate the initial hidden unit
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len, col_len, self.cond_num_name_enc)
        num_col_att_val = self.cond_num_col_att(e_num_col).squeeze()
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -100
        num_col_att = self.softmax(num_col_att_val)
        K_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)
        cond_num_h1 = self.cond_num_col2hid1(K_num_col).view(B, 2*self.N_depth, self.N_h//2).transpose(0, 1).contiguous()
        cond_num_h2 = self.cond_num_col2hid2(K_num_col).view(B, 2*self.N_depth, self.N_h//2).transpose(0, 1).contiguous()

        # Then run the LSTM and predict condition number.
        h_num_enc, _ = run_lstm(self.cond_num_lstm, x_emb_var, x_len,
                                hidden=(cond_num_h1, cond_num_h2))

        num_att_val = self.cond_num_att(h_num_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -100
        num_att = self.softmax(num_att_val)

        K_cond_num = (h_num_enc * num_att.unsqueeze(2).expand_as(h_num_enc)).sum(1)
        cond_num_score = self.cond_num_out(K_cond_num)
        cond_num = torch.argmax(cond_num_score, dim=1)

        assert cond_num_score.shape == (B, 5)
        assert cond_num.shape[-1] == B

        # Predict the columns of conditions
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.cond_col_name_enc)
        h_col_enc, _ = run_lstm(self.cond_col_lstm, x_emb_var, x_len)

        col_att_val = torch.bmm(e_cond_col,
                self.cond_col_att(h_col_enc).transpose(1, 2))

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                col_att_val[idx, :, num:] = -100

        col_att = self.softmax(col_att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        K_cond_col = (h_col_enc.unsqueeze(1) * col_att.unsqueeze(3)).sum(2)

        cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col) +
                self.cond_col_out_col(e_cond_col)).squeeze()

        # padding -100 for different sequence length
        max_col_num = max(col_num)
        for b, num in enumerate(col_num):
            if num < max_col_num:
                cond_col_score[b, num:] = -100

        # use gt_cond when training, use predicted gt_cond when testing
        # chosen_col_gt = [[col_1, col_2, ...], ..., col chosen for last query]
        if gt_cond is None:
            # select num of condition from 0 to 4
            cond_nums = np.argmax(cond_num_score.data.cpu().numpy(), axis=1)
            col_scores = cond_col_score.data.cpu().numpy()

            # select the first cond_nums columns as prediction
            chosen_col_gt = [list(np.argsort(-col_scores[b])[:cond_nums[b]]) for b in range(len(cond_nums))]
        else:
            chosen_col_gt = [[x[0] for x in one_gt_cond] for one_gt_cond in gt_cond]

        # force max column index is under the table length
        assert len(chosen_col_gt) == B
        for b in range(B):
            table_length = len(table_content[b])
            for c_index, x in enumerate(chosen_col_gt[b]):
                if x > table_length - 1:
                    chosen_col_gt[b][c_index] = table_length - 1

        # Predict the operator of conditions
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                                        col_len, self.cond_op_name_enc)
        h_op_enc, _ = run_lstm(self.cond_op_lstm, x_emb_var, x_len)
        col_emb = []

        # Pad the columns to 4, stack([col_1_emb, col_2_emb, pad_0_emb, pad_0_emb])
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x] for x in chosen_col_gt[b]]
                                      + [e_cond_col[b, 0]] * (4 - len(chosen_col_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        op_att_val = torch.matmul(self.cond_op_att(h_op_enc).unsqueeze(1),
                                  col_emb.unsqueeze(3)).squeeze()

        # add pad for query embedding
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                op_att_val[idx, :, num:] = -100

        op_att = self.softmax(op_att_val.view(B*4, -1)).view(B, 4, -1)
        K_cond_op = (h_op_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2)

        cond_op_score = self.cond_op_out(self.cond_op_out_K(K_cond_op) +
                self.cond_op_out_col(col_emb)).squeeze()

        # cond_op = []
        # for b in range(B):
        #     cond_op.extend([cond_op_score[b, num] for num in range(cond_num[b])])
        # cond_op = torch.stack(cond_op)
        # print('cond_op.shape', cond_op.shape)
        # print('cond_num', cond_num)
        # cond_op = torch.argmax(cond_op, dim=-1)


        # cond_op_score=[batch size, max condition num for one query, operation num]
        assert cond_op_score.shape == (B, 4, 4)

        # Predict the value of conditions
        h_str_enc, _ = run_lstm(self.cond_str_lstm, x_emb_var, x_len)
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                                        col_len, self.cond_str_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x] for x in chosen_col_gt[b]]
                                      + [e_cond_col[b, 0]] * (4 - len(chosen_col_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        if self.use_table:
            # column_content = []
            # for b in range(B):
            #     column_content.append([table_content[b][x] for x in chosen_col_gt[b]])
            # content_emb, table_config = self.emb_layer.gen_table_batch(column_content)

            if gt_where is not None:
                assert len(gt_where) == 4
                gt_index, gt_value, condition_num, max_value_length = gt_where
                # condition num for one batch queries
                assert len(gt_value) == gt_index.shape[0]

                # gt_value_embed=[condition num, max value num, hidden state]
                gt_value_embd = self.emb_layer.condition_value_batch(gt_value, max_value_length)
                # print('gt_value_embd.shape', gt_value_embd.shape)

                query_embed = torch.mean(x_emb_var, dim=1)
                queries_embed = torch.zeros([len(gt_index), 768])
                one_cond = 0
                for c, cond in enumerate(condition_num):
                    for _ in range(cond):
                        queries_embed[one_cond, :] = query_embed[c]
                        one_cond +=1
                assert one_cond == gt_index.shape[0]
                # print('queries_embed.shape', queries_embed.shape)

                gt_value_embd = gt_value_embd.transpose(0, 1)
                value_score = torch.matmul(gt_value_embd, queries_embed.transpose(0, 1))
                value_score = torch.diagonal(value_score, offset=0, dim1=-2, dim2=-1).transpose(0, 1)
                # print('value_score.shape', value_score.shape)

                # value_score=[condition num, max value length]
                for l, length in enumerate([len(x) for x in gt_value]):
                    if length < max_value_length:
                        value_score[l, length:] = -100

                value_utils = self.softmax(value_score)

            else:
                cond_value, max_value_length = self.cond_value_batch(table_content, q_seq,
                                                                    cond_num, chosen_col_gt, cond_op_score)

                cond_value_embd = self.emb_layer.condition_value_batch(cond_value, max_value_length)

                query_embed = torch.mean(x_emb_var, dim=1)
                queries_embed = torch.zeros([len(cond_value), 768])

                one_cond = 0
                for c, cond in enumerate(cond_num):
                    for _ in range(cond):
                        queries_embed[one_cond, :] = query_embed[c]
                        one_cond += 1
                assert one_cond == len(cond_value)

                cond_value_embd = cond_value_embd.transpose(0, 1)
                value_score = torch.matmul(cond_value_embd, queries_embed.transpose(0, 1))
                value_score = torch.diagonal(value_score, offset=0, dim1=-2, dim2=-1).transpose(0, 1)
                for l, length in enumerate([len(x) for x in cond_value]):
                    if length < max_value_length:
                        value_score[l, length:] = -100

                softmax_score = self.softmax(value_score)
                value_utils = (softmax_score, cond_value, cond_num)

        else:
            # for training
            if gt_where is not None:
                # gt_where = [[[index range for value in query], column 2, ...],table 2, ...]
                gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where)
                g_str_s_flat, _ = self.cond_str_decoder(
                        gt_tok_seq.view(B*4, -1, self.max_tok_num))
                g_str_s = g_str_s_flat.contiguous().view(B, 4, -1, self.N_h)

                h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
                g_ext = g_str_s.unsqueeze(3)
                col_ext = col_emb.unsqueeze(2).unsqueeze(2)

                cond_str_score = self.cond_str_out(
                        self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext) +
                        self.cond_str_out_col(col_ext)).squeeze()

                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        # [batch_size, condition num, T, TOK_NUM]
                        cond_str_score[b, :, :, num:] = -100

            # for validation and inference
            else:
                h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
                col_ext = col_emb.unsqueeze(2).unsqueeze(2)
                scores = []

                t = 0
                init_inp = np.zeros((B*4, 1, self.max_tok_num), dtype=np.float32)
                # Set the <BEG> token
                init_inp[:,0,0] = 1
                if self.gpu:
                    cur_inp = torch.from_numpy(init_inp).to('cuda')
                else:
                    cur_inp = torch.from_numpy(init_inp)
                cur_h = None
                while t < 50:
                    if cur_h:
                        g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp, cur_h)
                    else:
                        g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp)
                    g_str_s = g_str_s_flat.view(B, 4, 1, self.N_h)
                    g_ext = g_str_s.unsqueeze(3)

                    cur_cond_str_score = self.cond_str_out(
                            self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext)
                            + self.cond_str_out_col(col_ext)).squeeze()
                    for b, num in enumerate(x_len):
                        if num < max_x_len:
                            cur_cond_str_score[b, :, num:] = -100
                    scores.append(cur_cond_str_score)

                    _, ans_tok_var = cur_cond_str_score.view(B*4, max_x_len).max(1)
                    ans_tok = ans_tok_var.data.cpu()
                    cur_inp = torch.zeros(B*4, self.max_tok_num).scatter_(
                            1, ans_tok.unsqueeze(1), 1)
                    if self.gpu:
                        cur_inp = cur_inp.to('cuda')

                    cur_inp = cur_inp.unsqueeze(1)

                    t += 1

                cond_str_score = torch.stack(scores, 2)
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cond_str_score[b, :, :, num:] = -100

        if self.use_table:
            cond_score = (cond_num_score, cond_col_score, cond_op_score, value_utils)
        else:
            cond_score = (cond_num_score, cond_col_score, cond_op_score, cond_str_score)

        return cond_score
