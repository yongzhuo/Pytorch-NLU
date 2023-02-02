
# [***pytorch-textsummary***](https://github.com/yongzhuo/Pytorch-NLU/pytorch_textsummary)
>>> pytorch-textsummary是一个以pytorch和transformers为基础，专注于中文文本摘要的轻量级自然语言处理工具，支持抽取式摘要等。


## 目录
* [数据](#数据)
* [使用方式](#使用方式)
* [paper](#paper)
* [参考](#参考)


## 项目地址
   - pytorch-textsummary: [https://github.com/yongzhuo/Pytorch-NLU/pytorch_textsummary](https://github.com/yongzhuo/Pytorch-NLU/pytorch_textsummary)
  
  
# 数据
## 数据来源
免责声明：以下数据集由公开渠道收集而成, 只做汇总说明; 科学研究、商用请联系原作者; 如有侵权, 请及时联系删除。
  * [chinese_abstractive_corpus](https://github.com/wonderfulsuccess/chinese_abstractive_corpus), 教育培训行业抽象式自动摘要中文语料：语料库收集了教育培训行业主流垂直媒体的历史文章（截止到2018年6月5日）大约24500条数据集。主要是为训练抽象式模型而整理，每条数据有summary(摘要)和text(正文)，两个字段，Summary字段均为作者标注。
  * [NLPCC2017-task3-Single Document Summarization](http://tcci.ccf.org.cn/conference/2017/taskdata.php), NLPCC 2017 task3 单文档摘要;
  * [A Large-Scale Chinese Long-text Extractive Summarization Corpus](http://icrc.hitsz.edu.cn/info/1037/1411.htm), 哈工大长文本摘要数据;
  * [LCSTS: A Large-Scale Chinese Short Text Summarization Dataset](http://icrc.hitsz.edu.cn/info/1037/1141.htm), 哈工大LCSTS短文本摘要数据;	
  * 生成式文本摘要可以用一些带标题的文章来训练;

## 数据格式
```
1. 文本摘要  (txt格式, 每行为一个json):

1.1 抽取式文本摘要格式:
{"label": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0], "text": ["针对现有法向量估值算法都只能适用于某一类特定形状模型的问题。", "提出三维点云模糊分类的法向量估值算法。", "利用模糊推理系统对模型的点云数据分类。", "根据点云在不同形状区域的分布情况和曲率变化给出模糊规则。", "将点云分成属于平滑形状区域、薄片形状区域和尖锐形状区域三类。", "每类点云对应给出特定的法向量估值算法。", "由于任意模型形状分布的差别。", "其点云数据经过模糊分类后调用相应的估值算法次数会有差别。", "因此采用牙齿模型点云数据验证了算法的可行性。", "经过与三种典型算法比较可以看出本算法估算准确、简单可行。"]}
{"label": [0, 0, 1, 1, 0, 0], "text": ["医院物联网是物联网技术在医疗行业应用的集中体现。", "在简单介绍医院物联网基本概念的基础上。", "结合物联网机制和医院的实际特点。", "探讨了适用于医院物联网的体系结构。", "并分析了构建中的关键技术。", "包括医院物联网的标准建设、中间件技术及嵌入式电子病历的研究与设计等。"]}

```


# 使用方式
  更多样例sample详情见test/ts
  - 1. 需要配置好预训练模型目录, 即变量 pretrained_model_dir、pretrained_model_name_or_path、idx等;
  - 2. 需要配置好自己的语料地址, 即字典 model_config["path_train"]、model_config["path_dev"]
  - 3. cd到该脚本目录下运行普通的命令行即可, 例如: python3 tet_ts_base_train.py , python3 tet_ts_base_pred.py
## 文本摘要(TS), Text-Summary
```bash
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
```


# paper
## 文本摘要(TS), Text-Summary
* BertSum:   [Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318.pdf)


# 参考
This library is inspired by and references following frameworks and papers.

* GPT2-NewsTitle:   [https://github.com/liucongg/GPT2-NewsTitle](https://github.com/liucongg/GPT2-NewsTitle)
* BertSum: [https://github.com/nlpyang/BertSum](https://github.com/nlpyang/BertSum)


# Reference
For citing this work, you can refer to the present GitHub project. For example, with BibTeX:
```
@software{Pytorch-NLU,
    url = {https://github.com/yongzhuo/Pytorch-NLU},
    author = {Yongzhuo Mo},
    title = {Pytorch-NLU},
    year = {2021}
    
```
*希望对你有所帮助!
