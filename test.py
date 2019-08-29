import torch
from sqlnet.utils import *
from sqlnet.model.sqlnet import SQLNet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu')
    parser.add_argument('--toy', action='store_true', help='Small batchsize for fast debugging.')
    parser.add_argument('--train_emb', action='store_true', help='Use trained word embedding for SQLNet.')
    parser.add_argument('--output_dir', type=str, default='./result/', help='Output path of prediction result')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='the overall learning rate of model')
    parser.add_argument('--use_bert', action='store_true', help='using Bert to replace word embedding')
    parser.add_argument('--bert_path', type=str, default='/content/drive/My Drive/pre_trained/')
    parser.add_argument('--use_table', action='store_true', help='Whether use table content to predict Condition value')
    args = parser.parse_args()

    if args.toy:
        use_small=True
        gpu=args.gpu
        batch_size=16
    else:
        use_small=False
        gpu=args.gpu
        batch_size=64

    dev_sql, dev_table, dev_db, test_sql, test_table, test_db = load_dataset(use_small=use_small, mode='test')

    if args.use_bert:
        n_word = 768
        model = SQLNet(N_word=n_word, gpu=gpu, bert_path=args.bert_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    else:
        n_word = 300
        word_emb = load_word_emb('data/char_embedding.json')
        model = SQLNet(N_word=n_word, gpu=gpu, word_emb=word_emb, trainable_emb=args.train_emb)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)

    model_path = '../drive/My Drive/saved_model/best_model'
    print("Loading from %s" % model_path)
    model.load_state_dict(torch.load(model_path))
    print("Loaded model from %s" % model_path)

    dev_acc = epoch_acc(model, batch_size, dev_sql, dev_table, dev_db)
    print('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))

    print("Start to predict test set")
    predict_test(model, batch_size, test_sql, test_table, args.output_dir)
    print("Output path of prediction result is %s" % args.output_dir)
