# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: 文本摘要, text-summary


# 适配linux
import platform
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
path_sys = os.path.join(path_root, "pytorch_nlu", "pytorch_textsummary")
sys.path.append(path_sys)
print(path_root)
print(path_sys)

from tsTools import get_current_time
from tsConfig import model_config
from tsRun import TextSummary


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
    path_corpus = os.path.join(path_root, "pytorch_nlu", "corpus", "text_summary", "maths_toy")
    path_train = os.path.join(path_corpus, "train.json")
    path_dev = os.path.join(path_corpus, "dev.json")

    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train
    model_config["path_dev"] = path_dev
    model_config["lr"] = 1e-5  # 测试语料, 可为None
    model_config["max_len"] = 256  # 测试语料, 可为None
    model_config["batch_size"] = 32  # 测试语料, 可为None
    model_config["loss_type"] = "SOFT_MARGIN_LOSS"  # 测试语料, 可为None
    model_config["is_dropout"] = True  #
    model_config["is_adv"] = False  # 测试语料, 可为None


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
    model_config["model_save_path"] = "../output/text_summary/model_{}".format(model_type[idx])
    model_config["model_type"] = model_type[idx]
    # main
    lc = TextSummary(model_config)
    lc.process()
    lc.train()


# shell
# nohup python  tcRun.py > tc.log 2>&1 &
# tail -n 1000  -f tc.log
# |myz|

