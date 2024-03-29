# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: choice, model_config可配置参数


# 适配linux
import platform
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
path_sys = os.path.join(path_root, "pytorch_nlu", "pytorch_textclassification")
sys.path.append(path_sys)
print(path_root)
print(path_sys)
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
    path_corpus = os.path.join(path_root, "pytorch_nlu", "corpus", "text_classification", "school")
    path_train = os.path.join(path_corpus, "train.json")
    path_dev = os.path.join(path_corpus, "dev.json")
    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train  # 训练模语料, 必须
    model_config["path_dev"] = path_dev      # 验证语料, 可为None
    model_config["path_tet"] = None          # 测试语料, 可为None

    ## 参数配置 choice
    model_config["lr"] = 3e-5  # 5e-5, 1e-5 # 学习率
    model_config["max_len"] = 128  # 最大文本长度, padding
    model_config["batch_size"] = 64  # 批尺寸
    model_config["warmup_steps"] = 1000  # 预热步数
    model_config["is_active"] = False  # fc是否加激活函数
    model_config["is_dropout"] = True  # 是否随机丢弃
    model_config["is_adv"] = True  # 是否对抗训练
    # model_config["len_rate"] = 0.01   # 参与训练数据的样本数比率(如win10下少量数据跑通)
    model_config["epochs"] = 16  # 训练轮次 # 21  # 32

    model_config["output_hidden_states"] = [0, 1, 2, 3]  # 输出多层 # [0, 1, 2, 3]  # [1, 2, 11, 12]  # [8, 9, 10, 11, 12]  # [0,1,  5,6,  11,12]  # [1, 3, 5]
    model_config["loss_type"] = "PRIOR_MARGIN_LOSS"  # 损失函数
    # model_config["loss_type"] = "FOCAL_LOSS"
    # model_config["loss_type"] = "PRIOR_MARGIN_LOSS"
    # model_config["loss_type"] = "MIX_focal_prior"
    # model_config["loss_type"] = "MIX_prior_bce"
    # model_config["loss_type"] = "MIX_focal_bce"
    # model_config["loss_type"] = "BCE_MULTI"
    # model_config["loss_type"] = "BCE_LOGITS"
    # model_config["loss_type"] = "CIRCLE_LOSS"
    # model_config["loss_type"] = "CB_LOSS"
    # model_config["loss_type"] = "DB_LOSS"
    model_config["label_sep"] = "|myz|"  # 多标签分类类别标签分隔符

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
    idx = 0  # 选择的预训练模型类型---model_type, 0为BERT,
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]
    # model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx] + "_" + str(get_current_time()))
    model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx])
    model_config["model_type"] = model_type[idx]
    model_config["ADDITIONAL_SPECIAL_TOKENS"] = ["＋","－", "＝", "：", "．", "（", "）", "≈", "％",
                                                 "∥", "＜", "＞", "⊙", "≌", "。"]  # 新增特殊字符
    # main
    lc = TextClassification(model_config)
    lc.process()
    lc.train()


# shell
# nohup python  tet_tc_base_multi_label_choice.py > tc.tet_tc_base_multi_label_choice.py.log 2>&1 &
# tail -n 1000  -f tc.tet_tc_base_multi_label_choice.py.log
# |myz|

