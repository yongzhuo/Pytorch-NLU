
# [***pytorch-sequencelabeling***](https://github.com/yongzhuo/Pytorch-NLU/pytorch_sequencelabeling)
>>> pytorch-sequencelabeling是一个支持softmax、crf、span、grid等模型，只依赖pytorch、transformers、tensorboardX和numpy，专注于序列标注的轻量级自然语言处理工具包。


## 目录
* [数据](#数据)
* [使用方式](#使用方式)
* [paper](#paper)
* [参考](#参考)
* [Reference](#Reference)


## 项目地址
   - pytorch-sequencelabeling: [https://github.com/yongzhuo/Pytorch-NLU/pytorch_sequencelabeling](https://github.com/yongzhuo/Pytorch-NLU/pytorch_sequencelabeling)
  
  
# 数据
## 数据来源
免责声明：以下数据集由公开渠道收集而成, 只做汇总说明; 科学研究、商用请联系原作者; 如有侵权, 请及时联系删除。
### 通用数据集
  * ***Corpus_China_People_Daily***, 由北京大学计算语言学研究所发布的《人民日报》标注语料库PFR, 来源为《人民日报》1998上半年, 2014年, 2015上半年-2016.1-2017.1-2018.1(新时代人民日报分词语料库NEPD)等的内容, 包括中文分词cws、词性标注pos、命名实体识别ner...等标注数据;
  * ***Corpus_CTBX***, 由宾夕法尼亚大学(UPenn)开发并通过语言数据联盟（LDC） 发布的中文句法树库(Chinese Treebank), 来源为新闻数据、新闻杂志、广播新闻、广播谈话节目、微博、论坛、聊天对话和电话数据等, 包括中文分词cws、词性标注pos、命名实体识别ner...等标注数据;
  * [***NER-Weibo***](https://github.com/hltcoe/golden-horse), 中国社交媒体（微博）命名实体识别数据集（Weibo-NER-2015）, 该语料库包含2013年11月至2014年12月期间从微博上采集的1890条信息, 有两个版本(weiboNER.conll和weiboNER_2nd_conll), 共1890样例, 3个标签;
  * [***NER-CLUE***](https://github.com/CLUEbenchmark/CLUENER2020), 中文细粒度命名实体识别(CLUE-NER-2020), CLUE筛选标注的THUCTC数据集(清华大学开源的新闻内容文本分类数据集), 共12091样例, 10个标签; 
  * [***NER-Literature***](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset), 中文文学章篇级实体识别数据集(Literature-NER-2017), 数据来源为网站上1000多篇中国文学文章过滤提取的726篇, 共29096样本, 7个标签;
  * [***NER-Resume***](https://github.com/jiesutd/LatticeLSTM), 中文简历实体识别数据集(Resume-NER-2018), 来源为新浪财经网关于上市公司的高级经理人的简历摘要数据, 共1027样例，8个标签。
  * [***NER-BosonN***](https://github.com/bosondata), 中文新闻实体识别数据集(Boson-NER-2012), 数据集BosonNLP_NER_6C, 新增时间/公司名/产品名等标签, 共2000样例, 6个标签; 
  * [***NER-MSRA***](http://sighan.cs.uchicago.edu/bakeoff2005/), 中文新闻实体识别数据集(MSRA-NER-2005), 由微软亚洲研究院(MSRA)发布, 共55289样例, 通用的有3个标签, 完整的有26个标签;
## 比赛数据
### 实体识别
  * [CCKS 2021 中文NLP地址要素解析](https://tianchi.aliyun.com/competition/entrance/531900/information), 地址要素解析是将地址文本拆分成独立语义的要素，并对这些要素进行类型识别的过程。地址要素解析与地址相关性共同构成了中文地址处理两大核心任务，具有很大的商业价值。标注数据集由训练集、验证集和测试集组成，整体标注数据大约2万条左右。地址数据通过抓取公开的地址信息（如黄页网站等）获得， 均通过众包标注生成。
  * [CCKS 2021 面向中文电子病历的医疗实体及事件抽取](https://www.biendata.xyz/competition/ccks_2021_clinic/data/), 词表及电子病历数据由医渡云（北京）技术有限公司编写，标注数据由医渡云公司组织专业的医学团队进行人工标注，仅限CCKS竞赛评测用。本次评测的训练数据有：1500条中文标注数据, 1000条中文非标注数据, 6个类别的6292个中文实体词词表;
  * [CCKS 2020 面向试验鉴定的命名实体识别任务](https://www.biendata.xyz/competition/ccks_2020_8/), 军事装备试验鉴定是指通过规范化的组织形式和试验活动，对被试对象进行全面考核并作出评价结论的国家最高检验行为，涵盖方法、技术、器件、武器系统、平台系统、体系、训练演习等领域，涉及面广、专业性强。包括试验要素、性能指标、系统组成、任务场景共4个类别。
  * [2020语言与智能技术竞赛：关系抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/31), 百度实体关系抽取数据集, 其中实体部分可作为NER任务, 数据集共包含 48个已定义好的schema和超过21万中文句子，其中包括17万训练集，2万验证集和2万测试集。
  * [2020 非结构化商业文本信息中隐私信息识别](https://www.datafountain.cn/competitions/472), 明略科技商业文本数据集，本赛题要求参赛者从提供的非结构化商业文本信息中识别出文本中所涉及到的隐私数据，包括但不限于：（1）公司或个人基本信息：账号、姓名、联系方式、地址等；（2）商业秘密：制造方法、工艺流程、产品名称、专利名称等。共2515个样例, 14个类别。
  * [2020 中文医学文本命名实体识别](https://www.biendata.xyz/competition/chip_2020_1/data/), 数据集是由北京大学计算语言学教育部重点实验室、郑州大学信息工程学院自然语言处理实验室、哈尔滨工业大学（深圳）、以及鹏城实验室人工智能研究中心智慧医疗课题组联合构建。总字数达到164万，包含26903个句子，1062个文件，平均每个文件的字数为2393。数据集包含504种常见的儿科疾病、7,085种身体部位、12,907种临床表现、4,354种医疗程序等九大类医学实体。
  * [2020 中药说明书实体识别挑战](https://tianchi.aliyun.com/competition/entrance/531824/information), 本次标注数据源来自中药药品说明书，共包含1997份去重后的药品说明书，其中1000份用于训练数据，500份用作初赛测试数据，剩余的497份用作复赛的测试数据，共定义了13类实体。
  * [CCKS 2019 医疗命名实体识别](https://www.biendata.xyz/competition/ccks_2019_1/), 本任务是CCKS围绕中文电子病历语义化开展的系列评测的一个延续，在CCKS 2017，2018医疗命名实体识别评测任务的基础上进行了延伸和拓展。包括疾病和诊断、检查、检验、手术、药物和解剖部位等6种类别。
  * [CCKS 2019 中文短文本的实体链指](https://www.biendata.xyz/competition/ccks_2019_el/), 百度实体链指数据集, 面向中文短文本的实体识别与链指，简称ERL（Entity Recognition and Linking），是NLP领域的基础任务之一，即对于给定的一个中文短文本（如搜索Query、微博、用户对话内容、文章标题等）识别出其中的实体，并与给定知识库中的对应实体进行关联。ERL整个过程包括实体识别和实体链指两个子任务。7万训练集、1万开发集、1万测评集。
  * [CCKS 2019 人物关系抽取](https://www.biendata.xyz/competition/ccks_2019_ipre/), 任务关系抽取数据集IPRE（Inter-Personal Relationship Extraction），重点关注人物之间的关系抽取研究。给定一组人物实体对和包含该实体对的句子，找出给定实体对在已知关系表中的关系。
  * [CCKS 2018 面向中文电子病历的命名实体识别](https://www.biendata.xyz/competition/CCKS2018_1/), 本任务由清华大学知识工程实验室及医渡云（北京）技术有限公司联合主办，是CCKS 2017 CNER评测的改进和完善。实体类型聚焦在症状，药物，手术三大类，症状类型进一步细化，划分为三种类别, 总共五种类别, 600个现病史文档作为训练集，200 -400份作为测试集;

  
## 数据格式
```
1. 序列标注 (COLLN格式, SPAN格式):
1.1 COLLN格式(文件以.conll结尾):
青 B-ORG
岛 I-ORG
海 I-ORG
牛 I-ORG
队 I-ORG
和 O
广 B-ORG
州 I-ORG
松 I-ORG
日 I-ORG
队 I-ORG
的 O
雨 O
中 O
之 O
战 O

1.2 SPAN格式(文件以.span结尾):
{"label": [{"type": "ORG", "ent": "市委", "pos": [10, 11]}, {"type": "PER", "ent": "张敬涛", "pos": [14, 16]}], "text": "去年十二月二十四日，市委书记张敬涛召集县市主要负责同志研究信访工作时，提出三问：『假如上访群众是我们的父母姐妹，你会用什么样的感情对待他们？"}
{"label": [{"type": "PER", "ent": "金大中", "pos": [5, 7]}], "text": "今年2月，金大中新政府成立后，社会舆论要求惩治对金融危机负有重大责任者。"}
{"label": [], "text": "与此同时，作者同一题材的长篇侦破小说《鱼孽》也出版发行。"}

```


# 使用方式
  更多样例sample详情见test目录，以及slRun.py文件
  - 1. 需要配置好预训练模型目录, 即变量 pretrained_model_dir、pretrained_model_name_or_path、idx等;
  - 2. 需要配置好自己的语料地址, 即字典 model_config["path_train"]、model_config["path_dev"]
  - 3. cd到该脚本目录下运行普通的命令行即可, 例如: python3 slRun.py , python3 tet_sl_base_crf.py
## 序列标注, sequence-labeling
```bash

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
