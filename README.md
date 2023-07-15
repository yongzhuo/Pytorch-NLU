# [Pytorch-NLU](https://github.com/yongzhuo/Pytorch-NLU) 
[![PyPI](https://img.shields.io/pypi/v/Pytorch-NLU)](https://pypi.org/project/Pytorch-NLU/)
[![Build Status](https://travis-ci.com/yongzhuo/Pytorch-NLU.svg?branch=master)](https://travis-ci.com/yongzhuo/Pytorch-NLU)
[![PyPI_downloads](https://img.shields.io/pypi/dm/Pytorch-NLU)](https://pypi.org/project/Pytorch-NLU/)
[![Stars](https://img.shields.io/github/stars/yongzhuo/Pytorch-NLU?style=social)](https://github.com/yongzhuo/Pytorch-NLU/stargazers)
[![Forks](https://img.shields.io/github/forks/yongzhuo/Pytorch-NLU.svg?style=social)](https://github.com/yongzhuo/Pytorch-NLU/network/members)
[![Join the chat at https://gitter.im/yongzhuo/Pytorch-NLU](https://badges.gitter.im/yongzhuo/Pytorch-NLU.svg)](https://gitter.im/yongzhuo/Pytorch-NLU?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
>>> Pytorch-NLU是一个只依赖pytorch、transformers、numpy、tensorboardX，专注于文本分类、序列标注、文本摘要的极简自然语言处理工具包。
支持BERT、ERNIE、ROBERTA、NEZHA、ALBERT、XLNET、ELECTRA、GPT-2、TinyBERT、XLM、T5等预训练模型;
支持BCE-Loss、Focal-Loss、Circle-Loss、Prior-Loss、Dice-Loss、LabelSmoothing等损失函数;
具有依赖轻量、代码简洁、注释详细、调试清晰、配置灵活、拓展方便、适配NLP等特性。


## 目录
* [安装](#安装)
* [数据](#数据)
* [使用方式](#使用方式)
* [paper](#paper)
* [参考](#参考)
* [实验](#实验)


# 安装 
```bash
pip install Pytorch-NLU

# 清华镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Pytorch-NLU
```


# 数据
## 数据来源
免责声明：以下数据集由公开渠道收集而成, 只做汇总说明; 科学研究、商用请联系原作者; 如有侵权, 请及时联系删除。
### 文本分类
  * [baidu_event_extract_2020](https://aistudio.baidu.com/aistudio/competition/detail/32?isFromCcf=true), 项目以 2020语言与智能技术竞赛：事件抽取任务中的数据作为[多分类标签的样例数据](https://github.com/percent4/keras_bert_multi_label_cls)，借助多标签分类模型来解决, 共13456个样本, 65个类别;
  * [AAPD-dataset](https://git.uwaterloo.ca/jimmylin/Castor-data/tree/master/datasets/AAPD),  数据集出现在论文-SGM: Sequence Generation Model for Multi-label Classification, 英文多标签分类语料, 共55840样本, 54个类别;
  * [toutiao-news](https://github.com/fate233/toutiao-multilevel-text-classfication-dataset), 今日头条新闻标题, 多标签分类语料, 约300w-语料, 1000+类别;
  * [unknow-data](https://github.com/FBI1314/textClassification/tree/master/multilabel_text_classfication/data), 来源未知, 多标签分类语料, 约22339语料, 7个类别;
  * [SMP2018中文人机对话技术评测（ECDT）](https://worksheets.codalab.org/worksheets/0x27203f932f8341b79841d50ce0fd684f/), SMP2018 中文人机对话技术评测（SMP2018-ECDT）比赛语料, 短文本意图识别语料, 多类分类, 共3069样本, 31个类别;
  * [文本分类语料库（复旦）语料](http://www.nlpir.org/wordpress/2017/10/02/%e6%96%87%e6%9c%ac%e5%88%86%e7%b1%bb%e8%af%ad%e6%96%99%e5%ba%93%ef%bc%88%e5%a4%8d%e6%97%a6%ef%bc%89%e6%b5%8b%e8%af%95%e8%af%ad%e6%96%99/), 复旦大学计算机信息与技术系国际数据库中心自然语言处理小组提供的新闻语料, 多类分类语料, 共9804篇文档，分为20个类别。
  * [MiningZhiDaoQACorpus](https://github.com/liuhuanyong/MiningZhiDaoQACorpus), 中国科学院软件研究所刘焕勇整理的问答语料, 百度知道问答语料, 可以把领域当作类别, 多类分类语料, 100w+样本, 共17个类别;
  * [THUCNEWS](http://thuctc.thunlp.org/), 清华大学自然语言处理实验室整理的语料, 新浪新闻RSS订阅频道2005-2011年间的历史数据筛选, 多类分类语料, 74w新闻文档, 14个类别;
  * [IFLYTEK](https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip), 科大讯飞开源的长文本分类语料, APP应用描述的标注数据，包含和日常生活相关的各类应用主题, 链接为CLUE, 共17333样例, 119个类别;
  * [TNEWS](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip), 今日头条提供的中文新闻标题分类语料, 数据集来自今日头条的新闻版块, 链接为CLUE, 共73360样例, 15个类别;
### 序列标注
  * ***Corpus_China_People_Daily***, 由北京大学计算语言学研究所发布的《人民日报》标注语料库PFR, 来源为《人民日报》1998上半年, 2014年, 2015上半年-2016.1-2017.1-2018.1(新时代人民日报分词语料库NEPD)等的内容, 包括中文分词cws、词性标注pos、命名实体识别ner...等标注数据;
  * ***Corpus_CTBX***, 由宾夕法尼亚大学(UPenn)开发并通过语言数据联盟（LDC） 发布的中文句法树库(Chinese Treebank), 来源为新闻数据、新闻杂志、广播新闻、广播谈话节目、微博、论坛、聊天对话和电话数据等, 包括中文分词cws、词性标注pos、命名实体识别ner...等标注数据;
  * [***NER-Weibo***](https://github.com/hltcoe/golden-horse), 中国社交媒体（微博）命名实体识别数据集（Weibo-NER-2015）, 该语料库包含2013年11月至2014年12月期间从微博上采集的1890条信息, 有两个版本(weiboNER.conll和weiboNER_2nd_conll), 共1890样例, 3个标签;
  * [***NER-CLUE***](https://github.com/CLUEbenchmark/CLUENER2020), 中文细粒度命名实体识别(CLUE-NER-2020), CLUE筛选标注的THUCTC数据集(清华大学开源的新闻内容文本分类数据集), 共12091样例, 10个标签; 
  * [***NER-Literature***](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset), 中文文学章篇级实体识别数据集(Literature-NER-2017), 数据来源为网站上1000多篇中国文学文章过滤提取的726篇, 共29096样本, 7个标签;
  * [***NER-Resume***](https://github.com/jiesutd/LatticeLSTM), 中文简历实体识别数据集(Resume-NER-2018), 来源为新浪财经网关于上市公司的高级经理人的简历摘要数据, 共1027样例，8个标签。
  * [***NER-BosonN***](https://github.com/bosondata), 中文新闻实体识别数据集(Boson-NER-2012), 数据集BosonNLP_NER_6C, 新增时间/公司名/产品名等标签, 共2000样例, 6个标签; 
  * [***NER-MSRA***](http://sighan.cs.uchicago.edu/bakeoff2005/), 中文新闻实体识别数据集(MSRA-NER-2005), 由微软亚洲研究院(MSRA)发布, 共55289样例, 通用的有3个标签, 完整的有26个标签;


  
## 数据格式
```
1. 文本分类  (txt格式, 每行为一个json):

多类分类格式:
{"text": "人站在地球上为什么没有头朝下的感觉", "label": "教育"}
{"text": "我的小baby", "label": "娱乐"}
{"text": "请问这起交通事故是谁的责任居多小车和摩托车发生事故在无红绿灯", "label": "娱乐"}

多标签分类格式:
{"label": "3|myz|5", "text": "课堂搞东西，没认真听"}
{"label": "3|myz|2", "text": "测验90-94.A-"}
{"label": "3|myz|2", "text": "长江作业未交"}

2. 序列标注 (txt格式, 每行为一个json):

SPAN格式如下:
{"label": [{"type": "ORG", "ent": "市委", "pos": [10, 11]}, {"type": "PER", "ent": "张敬涛", "pos": [14, 16]}], "text": "去年十二月二十四日，市委书记张敬涛召集县市主要负责同志研究信访工作时，提出三问：『假如上访群众是我们的父母姐妹，你会用什么样的感情对待他们？"}
{"label": [{"type": "PER", "ent": "金大中", "pos": [5, 7]}], "text": "今年2月，金大中新政府成立后，社会舆论要求惩治对金融危机负有重大责任者。"}
{"label": [], "text": "与此同时，作者同一题材的长篇侦破小说《鱼孽》也出版发行。"}

CONLL格式如下:
青 B-ORG
岛 I-ORG
海 I-ORG
牛 I-ORG
队 I-ORG
和 O


3. 文本摘要  (txt格式, 每行为一个json):

3.1 抽取式文本摘要格式:
{"label": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0], "text": ["针对现有法向量估值算法都只能适用于某一类特定形状模型的问题。", "提出三维点云模糊分类的法向量估值算法。", "利用模糊推理系统对模型的点云数据分类。", "根据点云在不同形状区域的分布情况和曲率变化给出模糊规则。", "将点云分成属于平滑形状区域、薄片形状区域和尖锐形状区域三类。", "每类点云对应给出特定的法向量估值算法。", "由于任意模型形状分布的差别。", "其点云数据经过模糊分类后调用相应的估值算法次数会有差别。", "因此采用牙齿模型点云数据验证了算法的可行性。", "经过与三种典型算法比较可以看出本算法估算准确、简单可行。"]}
{"label": [0, 0, 1, 1, 0, 0], "text": ["医院物联网是物联网技术在医疗行业应用的集中体现。", "在简单介绍医院物联网基本概念的基础上。", "结合物联网机制和医院的实际特点。", "探讨了适用于医院物联网的体系结构。", "并分析了构建中的关键技术。", "包括医院物联网的标准建设、中间件技术及嵌入式电子病历的研究与设计等。"]}


```


# 使用方式
  更多样例sample详情见/test目录
  - 1. (已默认/可配置)需要配置好预训练模型目录, 即变量 pretrained_model_dir、pretrained_model_name_or_path、idx等;
  - 2. (已默认/可配置, 即训练, 验证集<非必要>)需要配置好自己的语料地址, 即字典 model_config["path_train"]、model_config["path_dev"]
  - 3. (命令行/编辑工具直接run)cd到该脚本目录下运行普通的命令行即可, 例如: python3 slRun.py , python3 tcRun.py , python3 tet_tc_base_multi_label.py, python3 tet_sl_base_crf.py
  - 4. (注意)如果训练时候出现指标为零或者很低的情况, 大概率是学习率、损失函数配错了
## 文本分类(TC), text-classification
```bash
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: 多标签分类, 根据label是否有|myz|分隔符判断是多类分类, 还是多标签分类


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
    # 损失函数类型,
    # multi-class:  可选 None(BCE), BCE, BCE_LOGITS, MSE, FOCAL_LOSS, DICE_LOSS, LABEL_SMOOTH
    # multi-label:  SOFT_MARGIN_LOSS, PRIOR_MARGIN_LOSS, FOCAL_LOSS, CIRCLE_LOSS, DICE_LOSS, MIX_focal_prior, DB_LOSS, CB_LOSS等
    model_config["loss_type"] = "SOFT_MARGIN_LOSS"
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
    idx = 0  # 选择的预训练模型类型---model_type, 0为BERT,
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
```
## 序列标注(SL), sequence-labeling
```bash
 !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: 序列标注, 命名实体识别, CRF, 条件随机场


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


# 预训练模型目录, 本地win10默认只跑2步就评估保存模型
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
    "BERT_WWM":  "hfl/chinese-bert-wwm-ext",
    "ROBERTA":  "hfl/chinese-roberta-wwm-ext",
    "ALBERT":  "uer/albert-base-chinese-cluecorpussmall",
    "XLNET":  "hfl/chinese-xlnet-mid",
    "ERNIE":  "nghuyong/ernie-1.0-base-zh",
    # "ERNIE": "nghuyong/ernie-3.0-base-zh",
    "BERT":  "bert-base-chinese",
    # "BERT": "hfl/chinese-macbert-base",

}


if __name__ == "__main__":
    # 训练-验证语料地址, 可以只输入训练地址
    path_corpus = os.path.join(path_root, "pytorch_nlu", "corpus", "sequence_labeling", "ner_china_people_daily_1998_conll")
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
    model_config["task_type"] = "SL-CRF"     # 任务类型, "SL-SOFTMAX", "SL-CRF", "SL-SPAN"

    model_config["dense_lr"] = 1e-5  # 最后一层的学习率, CRF层学习率/全连接层学习率, 1e-5, 1e-4, 1e-3
    model_config["lr"] = 1e-5        # 学习率, 1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 4e-4
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

```
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
        "BERT_WWM":  "hfl/chinese-bert-wwm-ext",
        "ROBERTA":  "hfl/chinese-roberta-wwm-ext",
        "ALBERT":  "uer/albert-base-chinese-cluecorpussmall",
        "XLNET":  "hfl/chinese-xlnet-mid",
        "ERNIE":  "nghuyong/ernie-1.0-base-zh",
        # "ERNIE": "nghuyong/ernie-3.0-base-zh",
        "BERT":  "bert-base-chinese",
        # "BERT": "hfl/chinese-macbert-base",

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
## 文本分类(TC, text-classification)
* FastText:   [Bag of Tricks for Efﬁcient Text Classiﬁcation](https://arxiv.org/abs/1607.01759)
* TextCNN：   [Convolutional Neural Networks for Sentence Classiﬁcation](https://arxiv.org/abs/1408.5882)
* charCNN-kim：   [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
* charCNN-zhang:  [Character-level Convolutional Networks for Text Classiﬁcation](https://arxiv.org/pdf/1509.01626.pdf)
* TextRNN：   [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
* RCNN：      [Recurrent Convolutional Neural Networks for Text Classification](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)
* DCNN:       [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188)
* DPCNN:      [Deep Pyramid Convolutional Neural Networks for Text Categorization](https://www.aclweb.org/anthology/P17-1052)
* VDCNN:      [Very Deep Convolutional Networks](https://www.aclweb.org/anthology/E17-1104)
* CRNN:        [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630)
* DeepMoji:    [Using millions of emojio ccurrences to learn any-domain represent ations for detecting sentiment, emotion and sarcasm](https://arxiv.org/abs/1708.00524)
* SelfAttention: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* HAN: [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
* CapsuleNet: [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
* TextGCN:               [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)
* Transformer(encode or decode): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* Bert:                  [BERT: Pre-trainingofDeepBidirectionalTransformersfor LanguageUnderstanding]()
* ERNIE:                 [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223)
* Xlnet:                 [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
* Albert:                [ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS](https://arxiv.org/pdf/1909.11942.pdf)
* RoBERTa:               [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* ELECTRA:               [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)
* GPT-2:                 [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
* T5:                    [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)


## 序列标注(SL, sequence-labeling)
* Bi-LSTM-CRF:    [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991.pdf)
* Bi-LSTM-LAN:    [Hierarchically-Reﬁned Label Attention Network for Sequence Labeling](https://arxiv.org/abs/1908.08676v2)
* CNN-LSTM:       [End-to-endSequenceLabelingviaBi-directionalLSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354)
* DGCNN:          [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
* CRF:            [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)
* Biaffine-BER:   [Named Entity Recognition as Dependency Parsing](https://aclanthology.org/2020.acl-main.577/)
* Lattice-LSTM:   [Lattice LSTM：Chinese NER Using Lattice LSTM](https://arxiv.org/abs/1805.02023)
* WC-LSTM:        [WC-LSTM: An Encoding Strategy Based Word-Character LSTM for Chinese NER Lattice LSTM](https://aclanthology.org/N19-1247.pdf)
* Lexicon:        [Simple-Lexicon：Simplify the Usage of Lexicon in Chinese NER](https://arxiv.org/pdf/1908.05969.pdf)
* FLAT:           [FLAT: Chinese NER Using Flat-Lattice Transformer](https://arxiv.org/pdf/2004.11795.pdf)
* MRC:            [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476v2)
* 
## 文本摘要(TS, Text-Summary)
* BertSum:   [Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318.pdf)


# 参考
This library is inspired by and references following frameworks and papers.

* keras与tensorflow版本对应: [https://docs.floydhub.com/guides/environments/](https://docs.floydhub.com/guides/environments/)
* BERT-NER-Pytorch: [https://github.com/lonePatient/BERT-NER-Pytorch](https://github.com/lonePatient/BERT-NER-Pytorch)
* bert4keras:   [https://github.com/bojone/bert4keras](https://github.com/bojone/bert4keras)
* Kashgari: [https://github.com/BrikerMan/Kashgari](https://github.com/BrikerMan/Kashgari)
* fastNLP: [https://github.com/fastnlp/fastNLP](https://github.com/fastnlp/fastNLP)
* HanLP: [https://github.com/hankcs/HanLP](https://github.com/hankcs/HanLP)
* FGM: [【炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)
* GlobalPointer: [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://kexue.fm/archives/8373)
* GlobalPointer_pytorch: [https://github.com/gaohongkui/GlobalPointer_pytorch](https://github.com/gaohongkui/GlobalPointer_pytorch)
* pytorch-loss: [pytorch-loss](https://github.com/CoinCheung/pytorch-loss)
* PriorLoss: [通过互信息思想来缓解类别不平衡问题](https://spaces.ac.cn/archives/7615)
* CircleLoss: [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)
* FocalLoss: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
* CRF: [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
* scikit-learn: [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
* tqdm: [https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm)
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


# 实验
## corpus==[unknow-data](https://github.com/FBI1314/textClassification/tree/master/multilabel_text_classfication/data), pretrain-model==ernie-tiny, batch=32, lr=5e-5, epoch=21

### 总结
#### micro-微平均
```
              precision    recall  f1-score   support

   micro_avg     0.7920    0.7189    0.7537       466    MARGIN_LOSS
   micro_avg     0.6706    0.8519    0.7505       466    PRIOR-MARGIN_LOSS
   micro_avg     0.8258    0.6309    0.7153       466    FOCAL_LOSS【0.5, 2】
   micro_avg     0.7890    0.7382    0.7627       466    CIRCLE_LOSS
   micro_avg     0.7612    0.7661    0.7636       466    DICE_LOSS【直接学习F1?】
   micro_avg     0.8062    0.7232    0.7624       466    BCE
   micro_avg     0.7825    0.7103    0.7447       466    BCE-Logits
   micro_avg     0.7899    0.7017    0.7432       466    BCE-Smooth
   micro_avg     0.7235    0.8197    0.7686       466    (FOCAL_LOSS【0.5, 2】 + PRIOR-MARGIN_LOSS) / 2
```

#### macro-宏平均
```
              precision    recall  f1-score   support

   macro_avg     0.6198    0.5338    0.5641       466    MARGIN_LOSS
   macro_avg     0.5103    0.7200    0.5793       466    PRIOR-MARGIN_LOSS
   macro_avg     0.7655    0.4973    0.5721       466    FOCAL_LOSS【0.5, 2】
   macro_avg     0.6275    0.5235    0.5627       466    CIRCLE_LOSS
   macro_avg     0.4287    0.3918    0.4025       466    DICE_LOSS【直接学习F1?】
   macro_avg     0.6978    0.5158    0.5828       466    BCE
   macro_avg     0.6046    0.5123    0.5433       466    BCE-Logits
   macro_avg     0.6963    0.5012    0.5721       466    BCE-Smooth
   macro_avg     0.6033    0.6809    0.6369       466    (FOCAL_LOSS【0.5, 2】 + PRIOR-MARGIN_LOSS) / 2

```
   
micro_avg     0.7235    0.8197    0.7686       466    
macro_avg     0.6033    0.6809    0.6369       466    


### 1. batch=32, loss=MARGIN_LOSS, lr=5e-5, epoch=21,        【精确率高些】
```
              precision    recall  f1-score   support

           3     0.8102    0.7919    0.8009       221
           2     0.8030    0.8030    0.8030       132
           1     0.7333    0.4925    0.5893        67
           6     0.7143    0.5000    0.5882        10
           5     0.7778    0.4828    0.5957        29
           0     0.0000    0.0000    0.0000         4
           4     0.5000    0.6667    0.5714         3

   micro_avg     0.7920    0.7189    0.7537       466
   macro_avg     0.6198    0.5338    0.5641       466
weighted_avg     0.7841    0.7189    0.7454       466
```

### 2. batch=32, loss=PRIOR-MARGIN_LOSS, lr=5e-5, epoch=21,  【召回率高些】
```
              precision    recall  f1-score   support

           3     0.7279    0.8959    0.8032       221
           2     0.7039    0.9545    0.8103       132
           1     0.5897    0.6866    0.6345        67
           6     0.3333    0.5000    0.4000        10
           5     0.6296    0.5862    0.6071        29
           0     0.1875    0.7500    0.3000         4
           4     0.4000    0.6667    0.5000         3

   micro_avg     0.6706    0.8519    0.7505       466
   macro_avg     0.5103    0.7200    0.5793       466
weighted_avg     0.6799    0.8519    0.7538       466
```

### 3. batch=32, loss=FOCAL_LOSS【(0.5, 2)】, lr=5e-5, epoch=21, 【精确率超级高, 0.25效果会变差】
```
              precision    recall  f1-score   support

           3     0.8482    0.7330    0.7864       221
           2     0.8349    0.6894    0.7552       132
           1     0.7586    0.3284    0.4583        67
           6     0.6667    0.4000    0.5000        10
           5     0.7500    0.4138    0.5333        29
           0     1.0000    0.2500    0.4000         4
           4     0.5000    0.6667    0.5714         3

   micro_avg     0.8258    0.6309    0.7153       466
   macro_avg     0.7655    0.4973    0.5721       466
weighted_avg     0.8206    0.6309    0.7038       466
```

### 4. batch=32, loss=CIRCLE_LOSS【, lr=5e-5, epoch=21, 【效果很好, 精确率召回率相对比较均衡】
```
              precision    recall  f1-score   support

           3     0.8125    0.8235    0.8180       221
           2     0.7914    0.8333    0.8118       132
           1     0.7333    0.4925    0.5893        67
           6     0.6667    0.4000    0.5000        10
           5     0.7222    0.4483    0.5532        29
           0     0.0000    0.0000    0.0000         4
           4     0.6667    0.6667    0.6667         3

   micro_avg     0.7890    0.7382    0.7627       466
   macro_avg     0.6275    0.5235    0.5627       466
weighted_avg     0.7785    0.7382    0.7521       466
```

### 5. batch=32, loss=DICE_LOSS, lr=5e-5, epoch=21, 【F1指标比较高, 少样本数据学不到, 不稳定】
```
              precision    recall  f1-score   support

           3     0.7714    0.8552    0.8112       221
           2     0.7727    0.9015    0.8322       132
           1     0.7347    0.5373    0.6207        67
           6     0.0000    0.0000    0.0000        10
           5     0.7222    0.4483    0.5532        29
           0     0.0000    0.0000    0.0000         4
           4     0.0000    0.0000    0.0000         3

   micro_avg     0.7612    0.7661    0.7636       466
   macro_avg     0.4287    0.3918    0.4025       466
weighted_avg     0.7353    0.7661    0.7441       466
```

### 6. batch=32, loss=BCE, lr=5e-5, epoch=21,        【普通的居然意外的好呢】
```
              precision    recall  f1-score   support

           3     0.8136    0.8100    0.8118       221
           2     0.8029    0.8333    0.8178       132
           1     0.8235    0.4179    0.5545        67
           6     0.6667    0.4000    0.5000        10
           5     0.7778    0.4828    0.5957        29
           0     0.0000    0.0000    0.0000         4
           4     1.0000    0.6667    0.8000         3

   micro_avg     0.8062    0.7232    0.7624       466
   macro_avg     0.6978    0.5158    0.5828       466
weighted_avg     0.8009    0.7232    0.7493       466
```

### 7. batch=32, loss=BCE_LOGITS, lr=5e-5, epoch=21, 【torch.nn.BCEWithLogitsLoss】
```

              precision    recall  f1-score   support

           3     0.7973    0.8009    0.7991       221
           2     0.8000    0.7879    0.7939       132
           1     0.7317    0.4478    0.5556        67
           6     0.6667    0.4000    0.5000        10
           5     0.7368    0.4828    0.5833        29
           0     0.0000    0.0000    0.0000         4
           4     0.5000    0.6667    0.5714         3

   micro_avg     0.7825    0.7103    0.7447       466
   macro_avg     0.6046    0.5123    0.5433       466
weighted_avg     0.7733    0.7103    0.7344       466
```

### 8. batch=32, loss=LABEL_SMOOTH, lr=5e-5, epoch=21, 【BCE-Label-smooth】
```
              precision    recall  f1-score   support

           3     0.7945    0.7873    0.7909       221
           2     0.8120    0.8182    0.8151       132
           1     0.7027    0.3881    0.5000        67
           6     0.8000    0.4000    0.5333        10
           5     0.7647    0.4483    0.5652        29
           0     0.0000    0.0000    0.0000         4
           4     1.0000    0.6667    0.8000         3

   micro_avg     0.7899    0.7017    0.7432       466
   macro_avg     0.6963    0.5012    0.5721       466
weighted_avg     0.7790    0.7017    0.7296       466
```

### 9. batch=32, loss=FOCAL_LOSS + PRIOR-MARGIN_LOSS, lr=5e-5, epoch=21, 【这两个Loss混合，宏平均(macro-avg)效果居然意外的好呢！】
```
           【1/2】
              precision    recall  f1-score   support

           3     0.7640    0.8643    0.8110       221
           2     0.7205    0.8788    0.7918       132
           1     0.6620    0.7015    0.6812        67
           6     0.4167    0.5000    0.4545        10
           5     0.7600    0.6552    0.7037        29
           0     0.4000    0.5000    0.4444         4
           4     0.5000    0.6667    0.5714         3

   micro_avg     0.7235    0.8197    0.7686       466
   macro_avg     0.6033    0.6809    0.6369       466
weighted_avg     0.7245    0.8197    0.7679       466
           
           【调和平均数】
              precision    recall  f1-score   support

           3     0.8474    0.7285    0.7835       221
           2     0.8304    0.7045    0.7623       132
           1     0.8182    0.4030    0.5400        67
           6     0.8000    0.4000    0.5333        10
           5     0.7143    0.3448    0.4651        29
           0     1.0000    0.2500    0.4000         4
           4     0.6667    0.6667    0.6667         3

   micro_avg     0.8324    0.6395    0.7233       466
   macro_avg     0.8110    0.4996    0.5930       466
weighted_avg     0.8292    0.6395    0.7132       466

           【1/3 + 2/3-focal】
              precision    recall  f1-score   support

           3     0.7890    0.8462    0.8166       221
           2     0.7516    0.8939    0.8166       132
           1     0.6935    0.6418    0.6667        67
           6     0.3636    0.4000    0.3810        10
           5     0.6538    0.5862    0.6182        29
           0     0.4000    0.5000    0.4444         4
           4     0.5000    0.6667    0.5714         3

   micro_avg     0.7430    0.8004    0.7707       466
   macro_avg     0.5931    0.6478    0.6164       466
weighted_avg     0.7420    0.8004    0.7686       466

           【1/4-prior + 3/4-focal】
              precision    recall  f1-score   support

           3     0.7956    0.8100    0.8027       221
           2     0.7712    0.8939    0.8281       132
           1     0.6981    0.5522    0.6167        67
           6     0.6667    0.4000    0.5000        10
           5     0.7143    0.5172    0.6000        29
           0     0.3333    0.2500    0.2857         4
           4     0.5000    0.6667    0.5714         3

   micro_avg     0.7656    0.7639    0.7648       466
   macro_avg     0.6399    0.5843    0.6007       466
weighted_avg     0.7610    0.7639    0.7581       466

           【4/9-prior + 5/9-focal】
              precision    recall  f1-score   support

           3     0.7819    0.8597    0.8190       221
           2     0.7578    0.9242    0.8328       132
           1     0.6567    0.6567    0.6567        67
           6     0.5000    0.5000    0.5000        10
           5     0.6250    0.5172    0.5660        29
           0     0.2857    0.5000    0.3636         4
           4     0.5000    0.6667    0.5714         3

   micro_avg     0.7364    0.8155    0.7739       466
   macro_avg     0.5867    0.6607    0.6156       466
weighted_avg     0.7352    0.8155    0.7715       466
```


### 10. pretrain-model==bert, batch=32, loss=FOCAL_LOSS + PRIOR-MARGIN_LOSS, lr=3e-5, epoch=21, 【这两个Loss混合，宏平均(micro-avg)效果居然意外的好呢！】
```
              precision    recall  f1-score   support

           3     0.7787    0.8597    0.8172       221
           2     0.7580    0.9015    0.8235       132
           1     0.7414    0.6418    0.6880        67
           6     0.7143    0.5000    0.5882        10
           5     0.6400    0.5517    0.5926        29
           0     0.0000    0.0000    0.0000         4
           4     0.5000    0.6667    0.5714         3

   micro_avg     0.7560    0.8047    0.7796       466
   macro_avg     0.5903    0.5888    0.5830       466
weighted_avg     0.7490    0.8047    0.7729       466
```


