# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: 多类分类, 根据label是否有|myz|分隔符判断是多类分类, 还是多标签分类


# 适配linux
import platform
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
path_sys = os.path.join(path_root, "pytorch_nlu", "pytorch_textclassification")
sys.path.append(path_sys)
print(path_root)
# 分类下的引入, pytorch_textclassification
from tcTools import get_current_time
from tcRun import TextClassification
from tcConfig import model_config


# 预训练模型地址, 本地win10默认只跑2步就评估保存模型
if platform.system().lower() == 'windows':
    # pretrained_model_dir = "D:/pretrain_models/pytorch"
    pretrained_model_dir = "E:/DATA/bert-model/00_pytorch"
    evaluate_steps = 32  # 评估步数
    save_steps = 32  # 存储步数
else:
    pretrained_model_dir = "/pretrain_models/pytorch"
    evaluate_steps = 320  # 评估步数
    save_steps = 320  # 存储步数
    ee = 0


if __name__ == "__main__":
    # 训练-验证语料地址, 可以只输入训练地址
    path_corpus = os.path.join(path_root, "pytorch_nlu", "corpus", "text_classification", "tnews")
    path_train = os.path.join(path_corpus, "train.json")
    path_dev = os.path.join(path_corpus, "dev.json")
    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train  # 训练模语料, 必须
    model_config["path_dev"] = path_dev      # 验证语料, 可为None
    model_config["path_tet"] = None          # 测试语料, 可为None
    # multi-class:  可选 None(BCE), BCE, BCE_LOGITS, MSE, FOCAL_LOSS, DICE_LOSS, LABEL_SMOOTH;
    # multi-label:  SOFT_MARGIN_LOSS, PRIOR_MARGIN_LOSS, FOCAL_LOSS, CIRCLE_LOSS, DICE_LOSS, MIX_focal_prior, DB_LOSS, CB_LOSS等
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config["CUDA_VISIBLE_DEVICES"])

    # 预训练模型适配的class
    model_type = ["BERT", "ERNIE", "BERT_WWM", "ALBERT", "ROBERTA", "XLNET", "ELECTRA"]
    pretrained_model_name_or_path = {
        "BERT_WWM":  "hfl/chinese-bert-wwm-ext",
        "ROBERTA":  "hfl/chinese-roberta-wwm-ext",
        "ALBERT":  "uer/albert-base-chinese-cluecorpussmall",
        "XLNET":  "hfl/chinese-xlnet-mid",
        "ERNIE":  "nghuyong/ernie-1.0-base-zh",
        # "ERNIE": "nghuyong/ernie-3.0-base-zh",
        "BERT":  "bert-base-chinese",
        # "BERT": "hfl/chinese-macbert-base",

    }
    idx = 0  # 选择的预训练模型类型---model_type, 0为BERT, 1为ernie
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]
    # model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx] + "_" + str(get_current_time()))
    model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx])
    model_config["model_type"] = model_type[idx]
    # main
    lc = TextClassification(model_config)
    lc.process()
    lc.train()


# shell
# nohup python  tcRun.py > tc.log 2>&1 &
# tail -n 1000  -f tc.log
# |myz|

