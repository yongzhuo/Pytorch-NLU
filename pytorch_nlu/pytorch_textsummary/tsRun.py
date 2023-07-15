# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: main programing, "训练时候logger不需要考虑"


# 适配linux
from collections import Counter
from argparse import Namespace
import random
import copy
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
sys.path.append(path_root)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tsConfig import _TS_MODEL_BERTSUM
from tsTools import get_logger
from tsOffice import Office
from tsData import DataSet


class TextSummary:
    def __init__(self, config):
        self.config = Namespace(**config)
        self.logger = get_logger(self.config.model_save_path)
        self.l2i, self.i2l = {}, {}

    def process(self):
        """ 数据预处理, process """
        # 数据读取
        # 训练集/验证集划分
        if self.config.path_dev:
            self.train_corpus = DataSet(self.config, self.config.path_train, self.logger)
            self.dev_corpus = DataSet(self.config, self.config.path_dev, self.logger)
        else:  # 没有验证集的时候, 默认划分 4:1
            self.train_corpus = DataSet(self.config, self.config.path_train, self.logger)
            xs_train, ys_train = self.train_corpus.data_iter
            len_rate_8 = int(len(ys_train) * 0.8)
            xs_train, ys_train = xs_train[:len_rate_8], ys_train[:len_rate_8]
            xs_dev, ys_dev = xs_train[len_rate_8:], ys_train[len_rate_8:]
            self.train_corpus.data_iter = xs_train, ys_train
            self.dev_corpus = DataSet(self.config, None, self.logger)
            self.dev_corpus.data_iter = xs_dev, ys_dev
        self.tet_corpus = DataSet(self.config, self.config.path_tet, self.logger)
        self.logger.info("read_corpus_from_json ok!")
        # 参数更新
        self.config.len_corpus = self.train_corpus.len_corpus
        self.config.prior_count = self.train_corpus.prior_count
        self.config.prior = self.train_corpus.prior
        self.config.task_type = _TS_MODEL_BERTSUM

    def train(self, path_save=None):
        """ 初始化训练  """
        # 创建模型目录与储存超参信息
        if not os.path.exists(self.config.model_save_path):
            os.makedirs(self.config.model_save_path, exist_ok=True)
        # 训练
        self.office = Office(tokenizer=self.train_corpus.tokenizer,
                             train_corpus=self.train_corpus,
                             dev_corpus=self.dev_corpus,
                             tet_corpus=self.tet_corpus,
                             config=self.config,
                             logger=self.logger)
        # 加载训练好的模型
        if path_save and path_save.strip():
            try:
                self.office.load_model_state(path_save)
            except Exception as e:
                self.logger.info(str(e))
                self.office.load_model(path_save)
        # 训练
        self.office.train_model()

    def eval(self):
        """ 验证评估  """
        try:
            self.office.load_model_state()
        except Exception as e:
            self.logger.info(str(e))
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
    from tsConfig import model_config
    from tsTools import get_current_time

    # 预训练模型地址, 本地win10默认只跑2步就评估保存模型
    if platform.system().lower() == 'windows':
        pretrained_model_dir = "D:/DATA/bert-model/00_pytorch"
        evaluate_steps = 32  # 评估步数
        save_steps = 32  # 存储步数
    else:
        pretrained_model_dir = "/pretrain_models/pytorch"
        path_ernie = "/home/moyzh/pretrain_models/pytorch/ernie-tiny"
        evaluate_steps = 320  # 评估步数
        save_steps = 320  # 存储步数
        ee = 0
    # 训练-验证语料地址, 可以只输入训练地址
    path_corpus = path_root + "/corpus/text_summary/maths_toy"
    path_train = os.path.join(path_corpus, "train.json")
    path_dev = os.path.join(path_corpus, "dev.json")

    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train
    model_config["path_dev"] = path_dev
    model_config["lr"] = 5e-5  # 测试语料, 可为None
    model_config["max_len"] = 256  # 测试语料, 可为None
    model_config["batch_size"] = 32  # 测试语料, 可为None
    model_config["loss_type"] = "SOFT_MARGIN_LOSS"  # 测试语料, 可为None
    model_config["is_adv"] = False  # 测试语料, 可为None

    # 损失函数类型,
    # multi-class:  可选 None(BCE), BCE, BCE_LOGITS, MSE, FOCAL_LOSS, DICE_LOSS, LABEL_SMOOTH
    # multi-label:  SOFT_MARGIN_LOSS, PRIOR_MARGIN_LOSS, FOCAL_LOSS, CIRCLE_LOSS, DICE_LOSS等
    # model_config["loss_type"] = "SOFT_MARGIN_LOSS"
    # model_config["loss_type"] = "MIX"
    # model_config["loss_type"] = "SOFT_MARGIN_LOSS"

    # 预训练模型适配的class
    model_type = ["BERT", "ERNIE", "BERT_WWM", "ALBERT", "ROBERTA", "XLNET", "ELECTRA"]
    pretrained_model_name_or_path = {
        "BERT_WWM": "hfl/chinese-bert-wwm-ext",
        "ROBERTA": "hfl/chinese-roberta-wwm-ext",
        "ALBERT": "uer/albert-base-chinese-cluecorpussmall",
        "XLNET": "hfl/chinese-xlnet-mid",
        "ERNIE": "nghuyong/ernie-1.0-base-zh",
        # "ERNIE": "nghuyong/ernie-3.0-base-zh",
        "BERT": "bert-base-chinese",
        # "BERT": "hfl/chinese-macbert-base",

    }
    idx = 1  # 选择的预训练模型类型---model_type
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]
    model_config["model_save_path"] = "../output/text_summary/model_{}".format(model_type[idx])
    model_config["model_type"] = model_type[idx]

    model_config["ADDITIONAL_SPECIAL_TOKENS"] = ["＋","－", "＝", "：", "．", "（", "）", "≈", "％",
                                                 "∥", "＜", "＞", "⊙", "≌", "。"]  # 新增特殊字符
    # main
    lc = TextSummary(model_config)
    lc.process()
    lc.train()


# shell
# nohup python  tsRun.py > tc_multi_class.log 2>&1 &
# tail -n 1000  -f tc_multi_class.log
# |myz|

