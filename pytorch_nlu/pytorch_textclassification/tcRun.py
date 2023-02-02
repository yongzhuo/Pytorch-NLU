# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: main programing, "训练时候logger不需要考虑"


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
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
        self.corpus = Corpus(self.config, self.logger)
        # 训练集/验证集划分
        if self.config.path_dev:
            xs_dev, ys_dev = self.corpus.read_corpus_from_json(self.config.path_dev, keys=self.config.xy_keys, len_rate=self.config.len_rate)
            xs_train, ys_train = self.corpus.read_corpus_from_json(self.config.path_train, keys=self.config.xy_keys, len_rate=self.config.len_rate)
        else:  # 没有验证集的时候, 默认划分 4:1
            xs_train, ys_train = self.corpus.read_corpus_from_json(self.config.path_train, keys=self.config.xy_keys, len_rate=self.config.len_rate)
            len_rate_8 = int(len(ys_train) * 0.8)
            xs_dev, ys_dev = xs_train[len_rate_8:], ys_train[len_rate_8:]
            xs_train, ys_train = xs_train[:len_rate_8], ys_train[:len_rate_8]
        self.logger.info("read_corpus_from_json ok!")
        # 排序, 样本多的类别排在前面
        ys = dict(Counter(ys_train))
        ys.update(dict(Counter(ys_dev)))
        # ys_sort = sorted(ys.items(), key=lambda x: x[1], reverse=True)  # 按照label的频率
        ys_sort = sorted(ys.items(), key=lambda x: x[0], reverse=True)  # 按照label名排序, 保证语料不一样的时候, 排序也一样
        # 处理标签, 包含多标签的情况
        TASK_TYPE = _TC_MULTI_CLASS
        ys_sep = []
        for y, _ in ys_sort:
            y_sep = str(y).split(self.config.label_sep)  #
            for yi in y_sep:
                ys_sep.append(yi)
                if yi and yi not in self.corpus.l2i:
                    self.corpus.l2i[str(yi)] = len(self.corpus.l2i)
                    self.corpus.i2l[str(len(self.corpus.l2i)-1)] = yi
            if len(y_sep) > 1:  # 如果有一个样例标签大于1, 则认为是多标签分类
                TASK_TYPE = _TC_MULTI_LABEL
        # 类别先验分布
        prior_count = [count for key, count in ys_sort if key]
        if _TC_MULTI_LABEL == TASK_TYPE:  # 处理多标签的情况
            ys_sep_dict = dict(Counter(ys_sep))
            prior_count = [ys_sep_dict[self.corpus.i2l[str(i)]] for i in range(len(self.corpus.l2i))]
        len_corpus = sum(prior_count)
        prior = [max(1-pc/len_corpus, 1/len_corpus) for pc in prior_count]
        self.logger.info("prior-label-count: {}".format([(k,v) for k, v in enumerate(ys_sort)]))
        self.logger.info("prior-label-prior: {}".format(prior))
        self.logger.info("prior_count: {}".format(prior_count))
        # 参数更新
        self.config.len_corpus = len_corpus
        self.config.prior_count = prior_count
        self.config.prior = prior
        self.logger.info(self.corpus.l2i)
        self.config.num_labels = len(self.corpus.l2i)
        self.config.task_type = TASK_TYPE
        self.config.max_len = self.corpus.max_len
        self.config.l2i = self.corpus.l2i
        self.config.i2l = self.corpus.i2l
        # token 转 idx, 训练集/验证集
        random.shuffle(xs_train)  # shuffle扰动
        # xs_train = xs_train[:int(len(xs_train)*0.18*2)]  ### len_rate
        self.train_data = self.corpus.preprocess(xs_train, self.config.l2i, max_len=self.config.max_len, label_sep=self.config.label_sep)
        self.dev_data = self.corpus.preprocess(xs_dev, self.config.l2i, max_len=self.config.max_len, label_sep=self.config.label_sep)
        # 测试集
        xs_tet, ys_tet = self.corpus.read_corpus_from_json(self.config.path_tet, keys=self.config.xy_keys) if self.config.path_tet else ([], [])
        self.tet_data = self.corpus.preprocess(xs_tet, self.config.l2i, max_len=self.config.max_len, label_sep=self.config.label_sep) if self.config.path_tet else None
        self.logger.info("self.corpus.preprocess ok!")

    def train(self, path_save=None):
        """ 初始化训练  """
        # 创建模型目录与储存超参信息
        if not os.path.exists(self.config.model_save_path):
            os.makedirs(self.config.model_save_path, exist_ok=True)
        # 训练
        self.office = Office(tokenizer=self.corpus.tokenizer,
                             train_corpus=self.train_data,
                             dev_corpus=self.dev_data,
                             tet_corpus=self.tet_data,
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
    from tcConfig import model_config
    from tcTools import get_current_time

    # 预训练模型地址, 本地win10默认只跑2步就评估保存模型
    if platform.system().lower() == 'windows':
        pretrained_model_dir = "E:/DATA/bert-model/00_pytorch"
        evaluate_steps = 2  # 评估步数
        save_steps = 2  # 存储步数
    else:
        pretrained_model_dir = "/pretrain_models/pytorch"
        evaluate_steps = 320  # 评估步数
        save_steps = 320  # 存储步数
        ee = 0
    # 训练-验证语料地址, 可以只输入训练地址
    path_corpus = path_root + "/corpus/text_classification/school"
    # path_corpus = path_root + "/corpus/text_classification/org_tnews"
    path_train = os.path.join(path_corpus, "train.json")
    path_dev = os.path.join(path_corpus, "dev.json")
    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train
    model_config["path_dev"] = path_dev
    # 损失函数类型,
    # multi-class:  可选 None(BCE), BCE, BCE_LOGITS, MSE, FOCAL_LOSS, DICE_LOSS, LABEL_SMOOTH
    # multi-label:  SOFT_MARGIN_LOSS, PRIOR_MARGIN_LOSS, FOCAL_LOSS, CIRCLE_LOSS, DICE_LOSS等
    # model_config["loss_type"] = "SOFT_MARGIN_LOSS"
    model_config["loss_type"] = "MIX"

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
        # "BERT": pretrained_model_dir + "/mengzi-bert-base/",
    }
    idx = 0  # 选择的预训练模型类型---model_type
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]
    # model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx] + "_" + str(get_current_time()))
    model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx])
    model_config["model_type"] = model_type[idx]

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config["CUDA_VISIBLE_DEVICES"])

    # main
    lc = TextClassification(model_config)
    lc.process()
    lc.train()


# shell
# nohup python  tcRun.py > tc_multi_class.log 2>&1 &
# tail -n 1000  -f tc_multi_class.log
# |myz|

