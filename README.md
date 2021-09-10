# [Pytorch-NLU](https://github.com/yongzhuo/Pytorch-NLU) 
[![PyPI](https://img.shields.io/pypi/v/Pytorch-NLU)](https://pypi.org/project/Pytorch-NLU/)
[![Build Status](https://travis-ci.com/yongzhuo/Pytorch-NLU.svg?branch=master)](https://travis-ci.com/yongzhuo/Pytorch-NLU)
[![PyPI_downloads](https://img.shields.io/pypi/dm/Pytorch-NLU)](https://pypi.org/project/Pytorch-NLU/)
[![Stars](https://img.shields.io/github/stars/yongzhuo/Pytorch-NLU?style=social)](https://github.com/yongzhuo/Pytorch-NLU/stargazers)
[![Forks](https://img.shields.io/github/forks/yongzhuo/Pytorch-NLU.svg?style=social)](https://github.com/yongzhuo/Pytorch-NLU/network/members)
[![Join the chat at https://gitter.im/yongzhuo/Pytorch-NLU](https://badges.gitter.im/yongzhuo/Pytorch-NLU.svg)](https://gitter.im/yongzhuo/Pytorch-NLU?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
>>> Pytorch-NLU是一个只依赖pytorch、transformers、numpy、tensorboardX，专注于文本分类、序列标注的极简自然语言处理工具包。
支持BERT、ERNIE、ROBERTA、NEZHA、ALBERT、XLNET、ELECTRA、GPT-2、TinyBERT、XLM、T5等预训练模型;
支持BCE-Loss、Focal-Loss、Circle-Loss、Prior-Loss、Dice-Loss、LabelSmoothing等损失函数;
具有依赖轻量、代码简洁、注释详细、调试清晰、配置灵活、拓展方便、适配NLP等特性。


## 目录
* [安装](#安装)
* [数据](#数据)
* [使用方式](#使用方式)
* [paper](#paper)
* [参考](#参考)


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

```


# 使用方式
  更多样例sample详情见/test目录
  - 1. 需要配置好预训练模型目录, 即变量 pretrained_model_dir、pretrained_model_name_or_path、idx等;
  - 2. 需要配置好自己的语料地址, 即字典 model_config["path_train"]、model_config["path_dev"]
  - 3. cd到该脚本目录下运行普通的命令行即可, 例如: python3 slRun.py , python3 tcRun.py , python3 tet_tc_base_multi_label.py, python3 tet_sl_base_crf.py
  - 4. 如果训练时候出现指标为零或者很低的情况, 大概率是学习率、损失函数配错了
## 文本分类(TC), text-classification
```bash
# !/usr/bin/python
# -*- coding: utf-8 -*-
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
sys.path.append(os.path.join(path_root, "pytorch_textclassification"))
print(path_root)
# 分类下的引入, pytorch_textclassification
from tcTools import get_current_time
from tcRun import TextClassification
from tcConfig import model_config

evaluate_steps = 320  # 评估步数
save_steps = 320  # 存储步数
# pytorch预训练模型目录, 必填
pretrained_model_name_or_path = "bert-base-chinese"
# 训练-验证语料地址, 可以只输入训练地址
path_corpus = os.path.join(path_root, "corpus", "text_classification", "school")
path_train = os.path.join(path_corpus, "train.json")
path_dev = os.path.join(path_corpus, "dev.json")


if __name__ == "__main__":
 
    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train  # 训练模语料, 必须
    model_config["path_dev"] = path_dev      # 验证语料, 可为None
    model_config["path_tet"] = None          # 测试语料, 可为None
    # 损失函数类型,
    # multi-class:  可选 None(BCE), BCE, BCE_LOGITS, MSE, FOCAL_LOSS, DICE_LOSS, LABEL_SMOOTH
    # multi-label:  SOFT_MARGIN_LOSS, PRIOR_MARGIN_LOSS, FOCAL_LOSS, CIRCLE_LOSS, DICE_LOSS等
    model_config["path_tet"] = "FOCAL_LOSS"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config["CUDA_VISIBLE_DEVICES"])

    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path
    model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx])
    model_config["model_type"] = "BERT"
    # main
    lc = TextClassification(model_config)
    lc.process()
    lc.train()

```
## 序列标注(SL), sequence-labeling
```bash

# 适配linux
import platform
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
path_sys = os.path.join(path_root, "pytorch_sequencelabeling")
sys.path.append(path_sys)
print(path_root)
print(path_sys)
# 分类下的引入, pytorch_textclassification
from slTools import get_current_time
from slRun import SequenceLabeling
from slConfig import model_config

evaluate_steps = 320  # 评估步数
save_steps = 320  # 存储步数
# pytorch预训练模型目录, 必填
pretrained_model_name_or_path = "bert-base-chinese"
# 训练-验证语料地址, 可以只输入训练地址
path_corpus = os.path.join(path_root, "corpus", "sequence_labeling", "ner_china_people_daily_1998_conll")
path_train = os.path.join(path_corpus, "train.conll")
path_dev = os.path.join(path_corpus, "dev.conll")


if __name__ == "__main__":
 
    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train  # 训练模语料, 必须
    model_config["path_dev"] = path_dev      # 验证语料, 可为None
    model_config["path_tet"] = None          # 测试语料, 可为None
    # 一种格式 文件以.conll结尾, 或者corpus_type=="DATA-CONLL"
    # 另一种格式 文件以.span结尾, 或者corpus_type=="DATA-SPAN"
    model_config["corpus_type"] = "DATA-CONLL"# 语料数据格式, "DATA-CONLL", "DATA-SPAN"
    model_config["task_type"] = "SL-CRF"     # 任务类型, "SL-SOFTMAX", "SL-CRF", "SL-SPAN"

    model_config["dense_lr"] = 1e-3  # 最后一层的学习率, CRF层学习率/全连接层学习率, 1e-5, 1e-4, 1e-3
    model_config["lr"] = 1e-5        # 学习率, 1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 4e-4
    model_config["max_len"] = 156    # 最大文本长度, None和-1则为自动获取覆盖0.95数据的文本长度, 0则取训练语料的最大长度, 具体的数值就是强制padding到max_len

    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path
    model_config["model_save_path"] = "../output/sequence_labeling/model_{}".format(model_type[idx])
    model_config["model_type"] = model_type[idx]
    # main
    lc = SequenceLabeling(model_config)
    lc.process()
    lc.train()

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

