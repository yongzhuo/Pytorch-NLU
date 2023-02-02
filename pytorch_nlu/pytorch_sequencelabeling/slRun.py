# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: main programing


# 适配linux
import platform
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(path_root)

from slConfig import model_config
os.environ["CUDA_VISIBLE_DEVICES"] = model_config.get("CUDA_VISIBLE_DEVICES", "0")
from slConfig import _SL_MODEL_SOFTMAX, _SL_MODEL_GRID, _SL_MODEL_SPAN, _SL_MODEL_CRF
from slConfig import _SL_DATA_CONLL, _SL_DATA_SPAN
from slTools import get_logger
from slOffice import Office
from slData import Corpus

from collections import Counter
from argparse import Namespace
import random
import copy


class SequenceLabeling:
    def __init__(self, config):
        self.config = Namespace(**config)
        self.logger = get_logger(self.config.model_save_path)
        # self.l2i, self.i2l = {}, {}

    def process(self):
        """ 数据预处理, process """
        corpus = Corpus(self.config, self.logger)
        # 数据读取, 确定数据类型
        if self.config.path_train.endswith(".conll") or self.config.corpus_type == _SL_DATA_CONLL:
            read_corpus = corpus.read_corpus_from_conll
            self.config.corpus_type = _SL_DATA_CONLL
        elif self.config.path_train.endswith(".span") or self.config.corpus_type == _SL_DATA_SPAN:
            read_corpus = corpus.read_corpus_from_span
            self.config.corpus_type = _SL_DATA_SPAN
        else:
            raise ValueError("invalid path of corpus: {}, must endswith '.conll' or '.json'".format(self.config.path_train))
        # 数据转化, 转化成输入训练的数据格式, SL-SPAN, SL-CRF, SL-SOFTMAX, _SL_MODEL_GRID
        if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF]:
            sl_preprocess = corpus.preprocess_common
        elif self.config.task_type.upper() in [_SL_MODEL_GRID]:
            sl_preprocess = corpus.preprocess_grid
        else:
            sl_preprocess = corpus.preprocess_span
        # 训练集/验证集划分
        if self.config.path_dev:
            xs_dev, ys_dev = read_corpus(self.config.path_dev, keys=self.config.xy_keys)
            # train放在后面, 方便后边获取先验知识ner-label-counter(prior)
            xs_train, ys_train = read_corpus(self.config.path_train, keys=self.config.xy_keys)
        else:  # 没有验证集的时候, 默认划分 4:1
            xs_train, ys_train = read_corpus(self.config.path_train, keys=self.config.xy_keys)
            len_rate_8 = int(len(ys_train) * 0.8)
            xs_dev, ys_dev = xs_train[len_rate_8:], ys_train[len_rate_8:]
            xs_train, ys_train = xs_train[:len_rate_8], ys_train[:len_rate_8]
        ### 本地只取部分数据测试（32条）
        if platform.system().lower() == "windows":
            xs_dev, ys_dev = xs_dev[:32], ys_dev[:32]
        self.logger.info("read_corpus_from_json ok!")
        # 排序, 样本多的类别排在前面
        if self.config.corpus_type == _SL_DATA_CONLL:
            ys = dict(Counter(ys_train[0]))
            for ysti in ys_train[1:] + ys_dev:
                ys.update(dict(Counter(ysti)))
            ys_sort = sorted(ys.items(), key=lambda x:x[1], reverse=True)
        elif self.config.corpus_type == _SL_DATA_SPAN:
            ys_type = []
            for ystd in ys_train+ys_dev:
                ys_type += [ysti.get("type", "") for ysti in ystd if ysti]
            ys = dict(Counter(ys_type))
            ys_sort = sorted(ys.items(), key=lambda x: x[1], reverse=True)
        else:
            raise ValueError("invalid line of data type")

        # 处理标签, S-city,
        # SOFTMAX、CRF、SPAN的情况下需要加 "O", SPAN不能放0, 必须从1开始, eg. start_id = [0,0,0,2,0,0]
        if self.config.task_type.upper() not in [_SL_MODEL_GRID]:
            corpus.l2i["O"] = 0
            corpus.i2l[str(0)] = "O"
        for y, _ in ys_sort:
            # "SL-SPAN" 只保存 Label-Type, 其他保存BIO/BIOES(即BILOU-BMEWO-)/BMES/IOB(即IOB-1)/
            if self.config.task_type.upper() in [_SL_MODEL_SPAN, _SL_MODEL_GRID]:
                y = y.split("-")[-1]
            if y not in corpus.l2i and "O" != y:
                corpus.l2i[str(y)] = len(corpus.l2i)
                corpus.i2l[str(len(corpus.l2i) - 1)] = y
        # 类别先验分布
        prior_count = [count for key, count in ys_sort]
        len_corpus = sum(prior_count)
        prior = [pc / len_corpus for pc in prior_count]
        self.logger.info("prior-label: {}".format(prior))
        # 参数更新
        self.config.num_labels = len(corpus.l2i)
        self.config.max_len = corpus.len_max
        self.config.l2i = corpus.l2i
        self.config.i2l = corpus.i2l
        self.config.prior = prior
        self.logger.info(corpus.l2i)
        # 重构BIO字典, span转conll的时候用得上
        l2i_conll = {"O": 0}
        for k, v in self.config.l2i.items():
            for st in self.config.sl_ctype:
                if st != "O":
                    l2i_conll[st + "-" + k] = len(l2i_conll)
        # token 转 idx, 训练集/验证集
        random.shuffle(xs_train)  # shuffle扰动
        self.train_data = sl_preprocess(xs_train, self.config.l2i, max_len=self.config.max_len, sl_ctype=self.config.sl_ctype, l2i_conll=l2i_conll)
        self.dev_data = sl_preprocess(xs_dev, self.config.l2i, max_len=self.config.max_len, sl_ctype=self.config.sl_ctype, l2i_conll=l2i_conll)
        # 测试集
        xs_tet, ys_tet = read_corpus(self.config.path_tet, keys=self.config.xy_keys) if self.config.path_tet else ([], [])
        self.tet_data = sl_preprocess(xs_tet, self.config.l2i, max_len=self.config.max_len, sl_ctype=self.config.sl_ctype, l2i_conll=l2i_conll) if self.config.path_tet else None
        if self.config.sl_ctype and self.config.corpus_type in [_SL_DATA_SPAN] and self.config.task_type in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF]:
            self.config.num_labels = len(l2i_conll)
            self.config.l2i_conll = copy.deepcopy(self.config.l2i)
            self.config.l2i = copy.deepcopy(l2i_conll)  # 重新赋值l2i, span转conll
            self.config.i2l = {str(i):k for i,k in enumerate(self.config.l2i)}  # conll类型
        self.logger.info("corpus.preprocess ok!")

    def train(self):
        """ 训练  """
        # 创建模型目录与储存超参信息
        if not os.path.exists(self.config.model_save_path):
            os.makedirs(self.config.model_save_path, exist_ok=True)
        # 训练
        self.office = Office(train_corpus=self.train_data,
                             dev_corpus=self.dev_data,
                             tet_corpus=self.tet_data,
                             config=self.config,
                             logger=self.logger)
        global_steps, best_mertics, best_report = self.office.train_model()
        return best_mertics

    def eval(self):
        """ 验证评估  """
        self.office.load_model()
        tet_results = self.office.evaluate("tet")
        return tet_results


if __name__ == "__main__":
    # 适配linux
    import sys
    import os
    path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(path_root)
    print(path_root)
    # 分类下的引入, pytorch_textclassification
    from slConfig import model_config
    from slTools import get_current_time
    # 预训练模型地址
    if platform.system().lower() == 'windows':
        # pretrained_model_dir = "D:/pretrain_models/pytorch"
        pretrained_model_dir = "E:/DATA/bert-model/00_pytorch"
        evaluate_steps = 2  # 评估步数
        save_steps = 2  # 存储步数
    else:
        pretrained_model_dir = "/pretrain_models/pytorch"
        evaluate_steps = 320  # 评估步数
        save_steps = 320  # 存储步数
        ee = 0
    # 训练-验证语料地址, 可以只输入训练地址
    # path_root = path_root + "/corpus/sequence_labeling/ner_china_people_daily_1998_span"
    # path_train = os.path.join(path_root, "train.span")
    # path_dev = os.path.join(path_root, "dev.span")
    path_root = path_root + "/corpus/sequence_labeling/ner_china_people_daily_1998_conll"
    path_train = os.path.join(path_root, "train.conll")
    path_dev = os.path.join(path_root, "dev.conll")
    model_config["path_train"] = path_train
    model_config["path_dev"] = path_dev
    model_config["path_tet"] = path_dev
    model_config["evaluate_steps"] = evaluate_steps   # 评估步数
    model_config["save_steps"] = save_steps   # 存储步数

    # 预训练模型适配的class
    model_type = ["BERT", "ERNIE", "BERT_WWM", "ALBERT", "ROBERTA", "XLNET", "ELECTRA"]
    pretrained_model_name_or_path = {
        "BERT_WWM": pretrained_model_dir + "/chinese_wwm_pytorch",
        "ROBERTA": pretrained_model_dir + "/chinese_roberta_wwm_ext_pytorch",
        "ALBERT": pretrained_model_dir + "/albert_base_v1",
        "XLNET": pretrained_model_dir + "/chinese_xlnet_mid_pytorch",
        "ERNIE": pretrained_model_dir + "/ERNIE_stable-1.0.1-pytorch",
        # "ERNIE": pretrained_model_dir + "/ernie-tiny",
        "BERT": pretrained_model_dir + "/bert-base-chinese",
    }
    idx = 0  # 选择的预训练模型类型---model_type
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]
    # model_config["model_save_path"] = "../output/sequence_labeling/model_{}".format(model_type[idx] + "_" + str(get_current_time()))
    model_config["model_save_path"] = "../output/sequence_labeling/model_{}".format(model_type[idx])
    model_config["model_type"] = model_type[idx]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config["CUDA_VISIBLE_DEVICES"])

    # main
    lc = SequenceLabeling(model_config)
    lc.process()
    lc.train()


# shell
# nohup python  slRun.py > sl_span.log 2>&1 &
# tail -n 1000  -f sl_span.log
# |myz|

