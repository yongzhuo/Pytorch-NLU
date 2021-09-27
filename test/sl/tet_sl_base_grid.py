# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/9/8 22:44
# @author  : Mo
# @function: 序列标注, 使用全局(网格)指针网络(GROBAL或GRID)架构的网络, eg.grid_id = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


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
    "BERT_WWM": pretrained_model_dir + "/chinese_wwm_pytorch",
    "ROBERTA": pretrained_model_dir + "/chinese_roberta_wwm_ext_pytorch",
    "ALBERT": pretrained_model_dir + "/albert_base_v1",
    "XLNET": pretrained_model_dir + "/chinese_xlnet_mid_pytorch",
    "ERNIE": pretrained_model_dir + "/ERNIE_stable-1.0.1-pytorch",
    # "ERNIE": pretrained_model_dir + "/ernie-tiny",  # 小模型
    "BERT": pretrained_model_dir + "/bert-base-chinese",
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
    model_config["task_type"] = "SL-GRID"  # 任务类型, "SL-SOFTMAX", "SL-CRF", "SL-SPAN"

    model_config["lr"] = 1e-5  # 学习率, 依据选择的预训练模型自己选择, 1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 4e-4
    model_config["dense_lr"] = 1e-5  # CRF层学习率/全连接层学习率, 1e-5, 1e-4, 1e-3
    model_config["max_len"] = 156    # 最大文本长度, None和-1则为自动获取覆盖0.95数据的文本长度, 0则取训练语料的最大长度, 具体的数值就是强制padding到max_len

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

