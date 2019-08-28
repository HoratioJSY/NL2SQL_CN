import json
from sqlnet.lib.dbengine import DBEngine
import numpy as np
from tqdm import tqdm
import re

# pattern = re.compile(r'[-一二三四五六七八九十百千万亿年\d]{2,}|\d+')


def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    for SQL_PATH in sql_paths:
        with open(SQL_PATH, encoding='utf-8') as inf:
            for idx, line in enumerate(inf):
                sql = json.loads(line.strip())
                if use_small and idx >= 10:
                    break
                sql_data.append(sql)
        print("Loaded %d data from %s" % (len(sql_data), SQL_PATH))

    for TABLE_PATH in table_paths:
        with open(TABLE_PATH, encoding='utf-8') as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab
        print("Loaded %d data from %s" % (len(table_data), TABLE_PATH))

    ret_sql_data = []
    for sql in sql_data:
        if sql[u'table_id'] in table_data:
            ret_sql_data.append(sql)

    return ret_sql_data, table_data


def load_dataset(use_small=False, mode='train'):
    print("Loading dataset")
    dev_sql, dev_table = load_data('data/val/val.json', 'data/val/val.tables.json', use_small=use_small)
    dev_db = 'data/val/val.db'
    if mode == 'train':
        train_sql, train_table = load_data('data/train/train.json', 'data/train/train.tables.json', use_small=use_small)
        train_db = 'data/train/train.db'
        return train_sql, train_table, train_db, dev_sql, dev_table, dev_db
    elif mode == 'test':
        test_sql, test_table = load_data('data/test/test.json', 'data/test/test.tables.json', use_small=use_small)
        test_db = 'data/test/test.db'
        return dev_sql, dev_table, dev_db, test_sql, test_table, test_db


def to_batch_seq(sql_data, table_data, idxes, st, ed, raw_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    gt_cond_seq = []
    raw_seq = []
    sel_num_seq = []
    table_content = []

    for i in range(st, ed):
        sql = sql_data[idxes[i]]

        # SELECT 子句选定 Column 的数量
        sel_num = len(sql['sql']['sel'])
        sel_num_seq.append(sel_num)

        # WHERE 子句选定 Condition 的数量，没参与模型训练？？？
        conds_num = len(sql['sql']['conds'])

        # 抽取问句，并做字符级的分割
        one_question = ''.join(sql['question'].split())
        q_seq.append([char for char in one_question])

        # 抽取 SQL 语句对应表格的表头，其中 table_data 曾单独取出 id 构造字典
        col_seq.append([[char for char in ''.join(header.split())] for header in table_data[sql['table_id']]['header']])

        # 抽取表格列数
        col_num.append(len(table_data[sql['table_id']]['header']))

        # 作为标注来计算模型损失，没有加入 WHERE value
        ans_seq.append(
            (
                len(sql['sql']['agg']),
                sql['sql']['sel'],
                sql['sql']['agg'],
                conds_num,
                # WHERE Column
                tuple(x[0] for x in sql['sql']['conds']),
                # WHERE Operator
                tuple(x[1] for x in sql['sql']['conds']),
                sql['sql']['cond_conn_op'],
            )
        )

        # 访问表格内容，忽略内容类型
        # table_content_types = table_data[sql['table_id']]['types']
        # table_content = [[[str1, str2, ...], column2, ...], table2, ...]
        one_table = []
        table_content_rows = table_data[sql['table_id']]['rows']
        for content_column in range(col_num[-1]):
            one_table.append([str(x[content_column]) for x in table_content_rows])
        table_content.append(one_table)

        # 另外用一个变量保存所有 WHERE Condition
        gt_cond_seq.append(sql['sql']['conds'])

        # 原始问题与表头
        raw_seq.append((sql['question'], table_data[sql['table_id']]['header']))
    if raw_data:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, raw_seq, table_content
    else:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, table_content


def to_batch_seq_test(sql_data, table_data, idxes, st, ed):
    q_seq = []
    col_seq = []
    col_num = []
    raw_seq = []
    table_ids = []
    table_content = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        one_question = ''.join(sql['question'].split())
        q_seq.append([char for char in one_question])
        col_seq.append([[char for char in ''.join(header.split())] for header in table_data[sql['table_id']]['header']])
        col_num.append(len(table_data[sql['table_id']]['header']))
        raw_seq.append(sql['question'])
        table_ids.append(sql['table_id'])

        one_table = []
        table_content_rows = table_data[table_ids[-1]]['rows']
        for content_column in range(col_num[-1]):
            one_table.append([x[content_column] for x in table_content_rows])
        table_content.append(one_table)
    return q_seq, col_seq, col_num, raw_seq, table_ids, table_content


def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        sql_data[idxes[i]]['sql']['conds'] = sql_data[idxes[i]]['sql']['conds']
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids


def cn_to_num(str, selected_num):
    digits_dict = {'零': '0', '一': '1', '二': "2", '两': '2', '三': '3', '四': '4',
                   '五': '5', '六': '6', '七': '7', '八': '8','九': '9'}
    num_dict = {'十': '0', '百': '00', '千': "000", '万': '0000', '亿': '00000000'}

    if re.search(r'年$', str) is not None:
        re_str = str
        for s in str:
            if digits_dict.__contains__(s):
                # replace_pattern = re.compile(r'%s' % s)
                re_str = re.sub(r'%s' % s, digits_dict[s], re_str)
            if num_dict.__contains__(s):
                re_str = re.sub(r'%s' % s, num_dict[s], re_str)

        if len(re_str) == 3 and int(re_str[0:-1]) > 50:
            final_str = '19' + re_str[0:-1]
        elif len(re_str) == 3:
            final_str = '20' + re_str[0:-1]
        else:
            final_str = re_str[0:-1]
    else:
        final_str = str
        for s in str:
            if digits_dict.__contains__(s):
                final_str = re.sub(r'%s' % s, digits_dict[s], final_str)
            if num_dict.__contains__(s):
                final_str = re.sub(r'%s' % s, num_dict[s], final_str)
                selected_num.extend(re.findall(r'[0-9]+', final_str))
    return final_str


def generate_gt_value(table, cond_seq, q):
    """
    :param table: all tables for one batch queries, all columns for one table
    :param cond_seq: [[[codition_coloumn, condition_type, condition_value],[...]], ..., ]
    :return:
            - gt_index: a tensor describe index of column and value, shape=[condition num, 2]
            - gt_value: all passable values for all conditions, [[value list for condition n], ...,]
    """
    pattern = re.compile(r'[两\-一二三四五六七八九十.百千万亿年\d]+')
    num_dict = {'十': '0', '百': '00', '千': "000", '万': '0000', '亿': '00000000'}
    gt_index = []
    gt_value = []
    condition_num = []
    # e_num = 0

    # len(cond_seq)=query_number
    for i, one_q_codition in enumerate(cond_seq):
        condition_num.append(len(one_q_codition))
        selected_num = pattern.findall(''.join(q[i]))
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

        selected_num.extend(re.findall(r'[0-9]+', ''.join(q[i])))
        selected_num = list(set(selected_num))

        selected_table = table[i]

        # len(one_q_condition) = num of condition for one query
        for one_condition in one_q_codition:
            gt_one_index = [one_condition[0]]
            selected_column = selected_table[one_condition[0]]
            for e, element in enumerate(selected_column):
                if re.search(".0$", element) is not None:
                    selected_column[e] = re.sub(r'.0$', '', element)
            # print(one_condition)
            try:
                # select column by ground truth
                if one_condition[1] >= 2:
                    # print('selected_column', selected_column)
                    gt_one_value = selected_column
                    gt_one_index.append(selected_column.index(one_condition[-1]))
                else:
                    # print('selected_num', selected_num)
                    gt_one_value = selected_num
                    gt_one_index.append(selected_num.index(one_condition[-1]))

            except BaseException as e:
                # print('==============================')
                # print('Bad case for condition value\'s ground truth: ', e)
                # print(''.join(q[i]))
                # print('selected_column', selected_column)
                # print('selected_num', selected_num)
                # e_num += 1
                gt_one_index.append(np.random.randint(0, len(gt_one_value), 1))
            gt_index.append(gt_one_index)
            gt_value.append(gt_one_value)
    max_value_length = max([len(x) for x in gt_value])
    assert np.array(condition_num).sum() == len(gt_value)

    return np.array(gt_index, dtype=np.int64), gt_value, condition_num, max_value_length


def epoch_train(model, optimizer, batch_size, sql_data, table_data, use_table=False):
    model.train()
    perm = np.random.permutation(len(sql_data))
    # perm = list(range(len(sql_data)))

    badcase = 0
    cum_loss = 0.0
    for st in tqdm(range(len(sql_data)//batch_size+1)):
        ed = (st+1)*batch_size if (st+1)*batch_size < len(perm) else len(perm)
        st = st * batch_size

        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, \
            table_content = to_batch_seq(sql_data, table_data, perm, st, ed)

        try:
            if use_table:
                gt_where_seq = generate_gt_value(table_content, gt_cond_seq, q_seq)
                # print(gt_index.shape)
                # print(len(gt_value))
                # print(np.array(condition_num).sum())
                # print(max_value_length)
                # quit()
            else:
                gt_where_seq = model.generate_gt_where_seq_test(q_seq, gt_cond_seq)
        except BaseException:
            badcase += 1
            print('badcase for generating gt_where_seq: ', badcase)
            continue

        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, table_content, gt_where=gt_where_seq, gt_cond=gt_cond_seq,
                              gt_sel=gt_sel_seq, gt_sel_num=gt_sel_num)
        # sel_num_score, sel_col_score, sel_agg_score, cond_score, cond_rela_score

        # compute loss
        loss = model.loss(score, ans_seq, gt_where_seq)
        cum_loss += loss.data.cpu().numpy()*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return cum_loss / len(sql_data)


def predict_test(model, batch_size, sql_data, table_data, output_path):
    model.eval()
    perm = list(range(len(sql_data)))
    fw = open(output_path, 'w')
    for st in tqdm(range(len(sql_data)//batch_size+1)):
        ed = (st+1)*batch_size if (st+1)*batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, col_seq, col_num, raw_q_seq, table_ids, table_content = to_batch_seq_test(sql_data, table_data, perm, st, ed)
        score = model.forward(q_seq, col_seq, col_num, table_content)
        sql_preds = model.gen_query(score, q_seq, col_seq, raw_q_seq)
        for sql_pred in sql_preds:
            sql_pred = eval(str(sql_pred))
            fw.writelines(json.dumps(sql_pred, ensure_ascii=False)+'\n')
            # fw.writelines(json.dumps(sql_pred,ensure_ascii=False).encode('utf-8')+'\n')
    fw.close()


def epoch_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)
    model.eval()
    perm = list(range(len(sql_data)))
    badcase = 0
    one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
    for st in tqdm(range(len(sql_data)//batch_size+1)):
        ed = (st+1)*batch_size if (st+1)*batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, raw_data, table_content = \
            to_batch_seq(sql_data, table_data, perm, st, ed, raw_data=True)

        # query_gt: ground truth of sql, data['sql'], containing sel, agg, conds:{sel, op, value}
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        raw_q_seq = [x[0] for x in raw_data]

        try:
            score = model.forward(q_seq, col_seq, col_num, table_content)
            # generate predicted format
            pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq)
            one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt)
        except:
            badcase += 1
            print('badcase for validation', badcase)
            continue

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        # Execution Accuracy
        for sql_gt, sql_pred, tid in zip(query_gt, pred_queries, table_ids):
            ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
            try:
                ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'], sql_pred['cond_conn_op'])
            except:
                ret_pred = None
            ex_acc_num += (ret_gt == ret_pred)
    return one_acc_num / len(sql_data), tot_acc_num / len(sql_data), ex_acc_num / len(sql_data)


def load_word_emb(file_name):
    print('Loading word embedding from %s'%file_name)
    f = open(file_name)
    ret = json.load(f)
    f.close()
    print('Vocabulary size: ', len(ret))
    return ret
