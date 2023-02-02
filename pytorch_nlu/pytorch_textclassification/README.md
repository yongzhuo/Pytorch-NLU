
# [***pytorch-textclassification***](https://github.com/yongzhuo/Pytorch-NLU/pytorch_textclassification)
>>> pytorch-textclassification是一个以pytorch和transformers为基础，专注于中文文本分类的轻量级自然语言处理工具，支持多类分类、多标签分类等。


## 目录
* [数据](#数据)
* [使用方式](#使用方式)
* [paper](#paper)
* [参考](#参考)


## 项目地址
   - pytorch-textclassification: [https://github.com/yongzhuo/Pytorch-NLU/pytorch_textclassification](https://github.com/yongzhuo/Pytorch-NLU/pytorch_textclassification)
  
  
# 数据
## 数据来源
免责声明：以下数据集由公开渠道收集而成, 只做汇总说明; 科学研究、商用请联系原作者; 如有侵权, 请及时联系删除。
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


## 数据格式
```
1. 文本分类  (txt格式, 每行为一个json):

1.1 多类分类格式:
{"text": "人站在地球上为什么没有头朝下的感觉", "label": "教育"}
{"text": "我的小baby", "label": "娱乐"}
{"text": "请问这起交通事故是谁的责任居多小车和摩托车发生事故在无红绿灯", "label": "娱乐"}

1.2 多标签分类格式:
{"label": "3|myz|5", "text": "课堂搞东西，没认真听"}
{"label": "3|myz|2", "text": "测验90-94.A-"}
{"label": "3|myz|2", "text": "长江作业未交"}

```


# 使用方式
  更多样例sample详情见test/tc目录
  - 1. 需要配置好预训练模型目录, 即变量 pretrained_model_dir、pretrained_model_name_or_path、idx等;
  - 2. 需要配置好自己的语料地址, 即字典 model_config["path_train"]、model_config["path_dev"]
  - 3. cd到该脚本目录下运行普通的命令行即可, 例如: python3 tcRun.py , python3 tet_tc_base_multi_label.py
## 文本分类(TC), Text-Classification
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
path_sys = os.path.join(path_root, "pytorch_nlu", "pytorch_textclassification")
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


# paper
## 文本分类(TC), Text-Classification
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
* Transformer(encode or decode): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* Bert:                  [BERT: Pre-trainingofDeepBidirectionalTransformersfor LanguageUnderstanding]()
* Xlnet:                 [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
* Albert:                [ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS](https://arxiv.org/pdf/1909.11942.pdf)
* RoBERTa:               [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* ELECTRA:               [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)
* TextGCN:               [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)


# 参考
This library is inspired by and references following frameworks and papers.

* keras与tensorflow版本对应: [https://docs.floydhub.com/guides/environments/](https://docs.floydhub.com/guides/environments/)
* bert4keras:   [https://github.com/bojone/bert4keras](https://github.com/bojone/bert4keras)
* Kashgari: [https://github.com/BrikerMan/Kashgari](https://github.com/BrikerMan/Kashgari)
* fastNLP: [https://github.com/fastnlp/fastNLP](https://github.com/fastnlp/fastNLP)
* HanLP: [https://github.com/hankcs/HanLP](https://github.com/hankcs/HanLP)
* FGM: [【炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)
* pytorch-loss: [pytorch-loss](https://github.com/CoinCheung/pytorch-loss)
* PriorLoss: [通过互信息思想来缓解类别不平衡问题](https://spaces.ac.cn/archives/7615)
* CircleLoss: [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)
* FocalLoss: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
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


希望对你有所帮助!

