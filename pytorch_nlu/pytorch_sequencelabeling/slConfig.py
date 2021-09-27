# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/8 20:42
# @author  : Mo
# @function: config of sequence-labeling, 超参数/类


import os
os.environ["USE_TORCH"] = "1"
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer, ElectraTokenizer, XLMTokenizer, AutoTokenizer
from transformers import BertConfig, RobertaConfig, AlbertConfig, XLNetConfig, ElectraConfig, XLMConfig, AutoConfig
from transformers import BertModel, RobertaModel, AlbertModel, XLNetModel, ElectraModel, XLMModel, AutoModel
# from transformers import LongformerTokenizer, LongformerConfig, LongformerModel
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model
from transformers import T5Tokenizer, T5Config, T5Model


PRETRAINED_MODEL_CLASSES = {
    # "LONGFORMER": (LongformerConfig, LongformerTokenizer, LongformerModel),
    "ELECTRA": (ElectraConfig, ElectraTokenizer, ElectraModel),
    "ROBERTA": (RobertaConfig, RobertaTokenizer, RobertaModel),
    "ALBERT": (AlbertConfig, AlbertTokenizer, AlbertModel),
    "XLNET": (XLNetConfig, XLNetTokenizer, XLNetModel),
    "ERNIE": (BertConfig, BertTokenizer, BertModel),
    "NEZHA": (BertConfig, BertTokenizer, BertModel),
    "BERT": (BertConfig, BertTokenizer, BertModel),
    "GPT2": (GPT2Config, GPT2Tokenizer, GPT2Model),
    "AUTO": (AutoConfig, AutoTokenizer, AutoModel),
    "XLM": (XLMConfig, XLMTokenizer, XLMModel),
    "T5": (T5Config, T5Tokenizer, T5Model)
}


# 标识符
_SL_MODEL_SOFTMAX = "SL-SOFTMAX"
_SL_MODEL_GRID = "SL-GRID"  # 网格, 即矩阵, Global-Pointer
_SL_MODEL_SPAN = "SL-SPAN"
_SL_MODEL_CRF = "SL-CRF"
_SL_DATA_CONLL = "DATA-CONLL"  # conll
_SL_DATA_SPAN = "DATA-SPAN"  # span


# model算法超参数
model_config = {
    "CUDA_VISIBLE_DEVICES": "1",  # 环境, GPU-CPU, "-1"/"0"/"1"/"2"...
    "output_hidden_states": None,  # 输出层, 即取第几层transformer的隐藏输出, list, eg. [6,11], None, [-1]
    "pretrained_model_name_or_path": "",  # 预训练模型地址
    "model_save_path": "model",  # 训练模型保存-训练完毕模型目录
    "config_name": "sl.config",  # 训练模型保存-超参数文件名
    "model_name": "sl.model",  # 训练模型保存-全量模型

    "path_train": None,  # 验证语料地址, 必传, string
    "path_dev": None,    # 验证语料地址, 必传, 可为None
    "path_tet": None,    # 验证语料地址, 必传, 可为None

    "corpus_type": "DATA-SPAN",  # 语料数据格式, "DATA-CONLL", "DATA-SPAN"
    "task_type": "SL-SPAN",  # 任务类型, "SL-SOFTMAX", "SL-CRF", "SL-SPAN", "SL-GRID", "sequence_labeling"
    "model_type": "BERT",   # 预训练模型类型, 如BERT/ROBERTA/ERNIE/ELECTRA/ALBERT
    "loss_type": "MARGIN_LOSS",  # 损失函数类型, 可选 None(BCE), BCE, MSE, FOCAL_LOSS,
                                 # multi-label:  MARGIN_LOSS, PRIOR_MARGIN_LOSS, CIRCLE_LOSS等
                                 # 备注: "SL-GRID"类型不要用BCE、PRIOR_MARGIN_LOSS
    "batch_size": 32,  # 批尺寸
    "num_labels": 0,   # 类别数, 自动更新
    "max_len": 128,    # 最大文本长度, None和-1则为自动获取覆盖0.95数据的文本长度, 0则取训练语料的最大长度, 具体的数值就是强制padding到max_len
    "epochs": 16,      # 训练轮次
    "dense_lr": 1e-5,  # CRF层学习率/全连接层学习率, CRF时候与lr保持100-1000倍的大小差距
    "lr": 1e-5,        # 学习率

    "grad_accum_steps": 1,  # 梯度积累多少步
    "max_grad_norm": 1.0,  # 最大标准化梯度
    "weight_decay": 0.99,  # lr学习率衰减系数
    "dropout_rate": 0.1,  # 随机失活概率
    "adam_eps": 1e-8,  # adam优化器超参
    "seed": 2021,  # 随机种子

    "stop_epochs": 4,  # 连续N轮无增长早停轮次
    "evaluate_steps": 320,  # 评估步数
    "save_steps": 320,  # 存储步数
    "warmup_steps": -1, # 预热步数, -1为取 0.5 的epoch步数
    "ignore_index": 0,  # 忽略的index
    "max_steps": -1,  # 最大步数, -1表示取满epochs
    "is_soft_label": True,  # 是否使用软标签, soft-label
    "is_train": True,  # 是否训练, 另外一个人不是(而是预测)
    "is_cuda": True,  # 是否使用gpu, 另外一个不是gpu(而是cpu)
    "is_adv": False,  # 是否使用对抗训练(默认FGM)
    "is_dropout": True,  # 最后几层输出是否使用随即失活
    "is_active": True,  # 最后几层输出是否使用激活函数, 如FCLayer/SpanLayer层
    "active_type": "GELU",  # 最后几层输出使用的激活函数, 可填写RELU/SIGMOID/TANH/MISH/SWISH/GELU

    "save_best_mertics_key": ["micro_avg", "f1-score"],  # 模型存储的判别指标, index-1可选: [micro_avg, macro_avg, weighted_avg],
                                                                          # index-2可选: [precision, recall, f1-score]
    "multi_label_threshold": 0.5,  # 多标签分类时候生效, 大于该阈值则认为预测对的
    "grid_pointer_threshold": 0,  # 网格(全局)指针网络阈值, 大于该阈值则认为预测对的
    "xy_keys_predict": ["text", "label"],  # 读取数据的格式, predict预测的时候用
    # "xy_keys": ["text", "label"],  # SPAN格式的数据, text, label在file中对应的keys
    "xy_keys": [0, 1],     # CONLL格式的数据, text, label在file中对应的keys, colln时候选择[0,1]等integer
    "label_sep": "|myz|",  # "|myz|" 多标签数据分割符, 用于多标签分类语料中
    "sl_ctype": "BIO",   #  数据格式sl-type, BIO, BMES, BIOES, 只在"corpus_type": "MYX", "task_type": "SL-CRL"或"SL-SOFTMAX"时候生效
    "head_size": 64,  # task_type=="SL-GRID"用

    # 是否对抗学习
    "adv_emb_name": "word_embeddings.",  # emb_name这个参数要换成你模型中embedding的参数名, model.embeddings.word_embeddings.weight
    "adv_eps": 1.0,  # 梯度权重epsilon

    "ADDITIONAL_SPECIAL_TOKENS": ["<macropodus>", "<macadam>"],  # 新增特殊字符
    "prior": None,  # 类别先验分布, 自动设置, 为一个label_num类别数个元素的list, json无法保存np.array
    "l2i_conll": None,
    "l2i": None,
    "i2l":None,
}

