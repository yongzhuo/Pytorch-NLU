# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: main programing


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(path_root)
from tcConfig import _TC_MULTI_CLASS, _TC_MULTI_LABEL
from tcTools import get_logger
from tcOffice import Office
from tcData import Corpus

from collections import Counter
from argparse import Namespace
import random


class TextClassification:
    def __init__(self, config):
        self.config = Namespace(**config)
        self.logger = get_logger(self.config.model_save_path)
        self.l2i, self.i2l = {}, {}

    def process(self):
        """ 数据预处理, process """
        # 数据读取
        corpus = Corpus(self.config, self.logger)
        # 训练集/验证集划分
        if self.config.path_dev:
            xs_dev, ys_dev = corpus.read_corpus_from_json(self.config.path_dev, keys=self.config.xy_keys)
            xs_train, ys_train = corpus.read_corpus_from_json(self.config.path_train, keys=self.config.xy_keys)
        else:  # 没有验证集的时候, 默认划分 4:1
            xs_train, ys_train = corpus.read_corpus_from_json(self.config.path_train, keys=self.config.xy_keys)
            len_rate_8 = int(len(ys_train) * 0.8)
            xs_dev, ys_dev = xs_train[len_rate_8:], ys_train[len_rate_8:]
            xs_train, ys_train = xs_train[:len_rate_8], ys_train[:len_rate_8]
        self.logger.info("read_corpus_from_json ok!")
        # 排序, 样本多的类别排在前面
        ys = dict(Counter(ys_train))
        ys.update(dict(Counter(ys_dev)))
        ys_sort = sorted(ys.items(), key=lambda x:x[1], reverse=True)
        # 处理标签, 包含多标签的情况
        TASK_TYPE = _TC_MULTI_CLASS
        ys_sep = []
        for y, _ in ys_sort:
            y_sep = str(y).split(self.config.label_sep)  #
            for yi in y_sep:
                ys_sep.append(yi)
                if yi not in corpus.l2i:
                    corpus.l2i[str(yi)] = len(corpus.l2i)
                    corpus.i2l[str(len(corpus.l2i)-1)] = yi
            if len(y_sep) > 1:
                TASK_TYPE = _TC_MULTI_LABEL
        # 类别先验分布
        prior_count = [count for key, count in ys_sort]
        if _TC_MULTI_LABEL == TASK_TYPE:  # 处理多标签的情况
            ys_sep_dict = dict(Counter(ys_sep))
            prior_count = [ys_sep_dict[corpus.i2l[str(i)]] for i in range(len(corpus.l2i))]
        len_corpus = sum(prior_count)
        prior = [pc / len_corpus for pc in prior_count]
        self.logger.info("prior-label: {}".format(prior))
        # 参数更新
        self.config.prior = prior
        self.logger.info(corpus.l2i)
        self.config.num_labels = len(corpus.l2i)
        self.config.task_type = TASK_TYPE
        self.config.max_len = corpus.len_max
        self.config.l2i = corpus.l2i
        self.config.i2l = corpus.i2l
        # token 转 idx, 训练集/验证集
        random.shuffle(xs_train)  # shuffle扰动
        self.train_data = corpus.preprocess(xs_train, self.config.l2i, max_len=self.config.max_len)
        self.dev_data = corpus.preprocess(xs_dev, self.config.l2i, max_len=self.config.max_len)
        # 测试集
        xs_tet, ys_tet = corpus.read_corpus_from_json(self.config.path_tet, keys=self.config.xy_keys) if self.config.path_tet else ([], [])
        self.tet_data = corpus.preprocess(xs_tet, self.config.l2i, max_len=self.config.max_len) if self.config.path_tet else None
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
        self.office.train_model()

    def eval(self):
        """ 验证评估  """
        self.office.load_model()
        tet_results = self.office.evaluate("tet")
        return tet_results


if __name__ == "__main__":
    # 适配linux
    import platform
    import sys
    import os
    path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(path_root)
    print(path_root)
    # 分类下的引入, pytorch_textclassification
    from tcConfig import model_config
    from tcTools import get_current_time
    # 预训练模型地址
    if platform.system().lower() == 'windows':
        pretrained_model_dir = "D:/pretrain_models/pytorch"
        evaluate_steps = 2  # 评估步数
        save_steps = 2  # 存储步数
    else:
        pretrained_model_dir = "/pretrain_models/pytorch"
        evaluate_steps = 320  # 评估步数
        save_steps = 320  # 存储步数
        ee = 0
    # 训练-验证语料地址, 可以只输入训练地址
    path_corpus = path_root + "/corpus/text_classification/school"
    path_train = os.path.join(path_corpus, "train.json")
    path_dev = os.path.join(path_corpus, "dev.json")
    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train
    model_config["path_dev"] = path_dev
    model_config["path_tet"] = path_dev

    # 预训练模型适配的class
    model_type = ["BERT", "ERNIE", "BERT_WWM", "ALBERT", "ROBERTA", "XLNET", "ELECTRA"]
    pretrained_model_name_or_path = {
        "BERT_WWM": pretrained_model_dir + "/chinese_wwm_pytorch",
        "ROBERTA": pretrained_model_dir + "/chinese_roberta_wwm_ext_pytorch",
        "ALBERT": pretrained_model_dir + "/albert_base_v1",
        "XLNET": pretrained_model_dir + "/chinese_xlnet_mid_pytorch",
        # "ERNIE": pretrained_model_dir + "/ERNIE_stable-1.0.1-pytorch",
        "ERNIE": pretrained_model_dir + "/ernie-tiny",
        "BERT": pretrained_model_dir + "/bert-base-chinese",
    }
    idx = 0  # 选择的预训练模型类型---model_type
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]
    # model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx] + "_" + str(get_current_time()))
    model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx])
    model_config["model_type"] = model_type[idx]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config["CUDA_VISIBLE_DEVICES"])

    # main
    lc = TextClassification(model_config)
    lc.process()
    lc.train()


# shell
# nohup python  tcRun.py > tc_multi.log 2>&1 &
# tail -n 1000  -f tc_multi.log
# |myz|

