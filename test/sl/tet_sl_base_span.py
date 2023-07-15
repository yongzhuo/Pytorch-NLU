# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: 序列标注, 使用SPAN架构的网络, 即start_id = [0,2,0,0.....]
#                                        end_id   = [0,0,0,2,0...]


# 适配linux
import platform
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
path_sys = os.path.join(path_root, "pytorch_nlu", "pytorch_sequencelabeling")
sys.path.append(path_sys)
print(path_root)
print(path_sys)
# 分类下的引入, pytorch_textclassification
from slTools import get_current_time
from slRun import SequenceLabeling
from slConfig import model_config


# 预训练模型地址, 本地win10默认只跑2步就评估保存模型
if platform.system().lower() == 'windows':
    pretrained_model_dir = "D:/pretrain_models/pytorch"
    evaluate_steps = 2  # 评估步数
    save_steps = 2  # 存储步数
else:
    pretrained_model_dir = "/pretrain_models/pytorch"
    evaluate_steps = 320  # 评估步数
    save_steps = 320  # 存储步数
    ee = 0


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


if __name__ == "__main__":
    # 训练-验证语料地址, 可以只输入训练地址
    path_corpus = os.path.join(path_root, "corpus", "sequence_labeling", "ner_china_people_daily_1998_conll")
    path_train = os.path.join(path_corpus, "train.conll")
    path_dev = os.path.join(path_corpus, "dev.conll")
    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train  # 训练模语料, 必须
    model_config["path_dev"] = path_dev      # 验证语料, 可为None
    model_config["path_tet"] = None          # 测试语料, 可为None
    # 一种格式 文件以.conll结尾, 或者corpus_type=="DATA-CONLL"
    # 另一种格式 文件以.span结尾, 或者corpus_type=="DATA-SPAN"
    model_config["corpus_type"] = "DATA-CONLL"# 语料数据格式, "DATA-CONLL", "DATA-SPAN"
    model_config["task_type"] = "SL-SPAN"  # 任务类型, "SL-SOFTMAX", "SL-CRF", "SL-SPAN"

    model_config["lr"] = 1e-5  # 学习率, 依据选择的预训练模型自己选择, 1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 4e-4
    model_config["dense_lr"] = 1e-5  # CRF层学习率/全连接层学习率, 1e-5, 1e-4, 1e-3
    idx = 0  # 选择的预训练模型类型---model_type, 0为BERT,
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]

    # model_config["model_save_path"] = "../output/sequence_labeling/model_{}".format(model_type[idx] + "_" + str(get_current_time()))
    model_config["model_save_path"] = "../output/sequence_labeling/model_{}".format(model_type[idx])
    model_config["model_type"] = model_type[idx]
    # main
    lc = SequenceLabeling(model_config)
    lc.process()
    lc.train()


# shell
# nohup python  slRun.py > sl.log 2>&1 &
# tail -n 1000  -f sl.log

