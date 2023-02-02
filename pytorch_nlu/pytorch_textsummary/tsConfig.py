# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/19 21:48
# @author  : Mo
# @function: config of transformers and graph-model


_TS_MODEL_BERTSUM = "TS_MODEL_BERTSUM"


# cuda设置
import platform
if platform.system().lower() == "windows":
    CUDA_VISIBLE_DEVICES = "0"
else:
    CUDA_VISIBLE_DEVICES = "0"


# model算法超参数
model_config = {
    "path_finetune": "",
    "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,  # 环境, GPU-CPU, "-1"/"0"/"1"/"2"...
    "USE_TORCH": "1",             # transformers使用torch, 因为脚本是torch写的
    "output_hidden_states": None,  # [6,11]  # 输出层, 即取第几层transformer的隐藏输出, list
    "pretrained_model_name_or_path": "",  # 预训练模型地址
    "model_save_path": "save_path",  # 训练模型保存-训练完毕模型目录
    "config_name": "tc.config",  # 训练模型保存-超参数文件名
    "model_name": "tc.model",  # 训练模型保存-全量模型
    "path_train": None,  # 验证语料地址, 必传, string
    "path_dev": None,  # 验证语料地址, 必传, 可为None
    "path_tet": None,  # 验证语料地址, 必传, 可为None

    "tokenizer_type": "BASE",  # tokenizer解析的类型, 默认transformers自带的, 可设"CHAR"(单个字符的, 不使用bpe等词根的)
    "task_type": _TS_MODEL_BERTSUM,  # 任务类型, 依据数据类型自动更新, "TS_MODEL_BERTSUM", TS为text-summary的缩写
    "model_type": "BERT",  # 预训练模型类型, 如bert, roberta, ernie
    "loss_type": "BCE",  # "BCE", # 损失函数类型,
                                  # multi-class:  可选 None(BCE), BCE, BCE_LOGITS, MSE, FOCAL_LOSS, DICE_LOSS, LABEL_SMOOTH, MIX;
                                  # multi-label:  SOFT_MARGIN_LOSS, PRIOR_MARGIN_LOSS, FOCAL_LOSS, CIRCLE_LOSS, DICE_LOSS, MIX等

    "batch_size": 32,  # 批尺寸
    "num_labels": 0,  # 类别数, 自动更新
    "max_len": 0,  # 最大文本长度, -1则为自动获取覆盖0.95数据的文本长度, 0为取得最大文本长度作为maxlen
    "epochs": 21,  # 训练轮次
    "lr": 1e-5,    # 学习率

    "grad_accum_steps": 1,  # 梯度积累多少步
    "max_grad_norm": 1.0,  # 最大标准化梯度
    "weight_decay": 0.99,  # lr衰减
    "dropout_rate": 0.1,  # 随即失活概率
    "adam_eps": 1e-8,  # adam优化器超参
    "seed": 2021,  # 随机种子, 3407, 2021

    "stop_epochs": 4,  # 早停轮次
    "evaluate_steps": 320,  # 评估步数
    "save_steps": 320,  # 存储步数
    "warmup_steps": -1,  # 预热步数
    "ignore_index": 0,  # 忽略的index
    "max_steps": -1,  # 最大步数, -1表示取满epochs
    "is_train": True,  # 是否训练, 另外一个人不是(而是预测)
    "is_cuda": True,  # 是否使用gpu, 另外一个不是gpu(而是cpu)
    "is_adv": False,  # 是否使用对抗训练(默认FGM)
    "is_dropout": True,  # 最后几层输出是否使用随即失活
    "is_active": True,  # 最后几层输出是否使用激活函数, 如FCLayer/SpanLayer层
    "active_type": "RELU",  # 最后几层输出使用的激活函数, 可填写RELU/SIGMOID/TANH/MISH/SWISH/GELU
    "is_fc_sigmoid": False,  # 最后一层是否使用sigmoid(训练时灵活配置, 存储模型时加上方便推理[如->onnx->tf-serving的时候])
    "is_fc_softmax": False,  # 最后一层是否使用softmax(训练时灵活配置, 存储模型时加上方便推理[如->onnx->tf-serving的时候])

    "save_best_mertics_key": ["micro_avg", "f1-score"],  # ["macro avg", "f1-score"],  # 模型存储的判别指标, index-1可选: [micro_avg, macro_avg, weighted_avg],
                                                                          # index-2可选: [precision, recall, f1-score]
    "multi_label_threshold": 0.5,  # 多标签分类时候生效, 大于该阈值则认为预测对的
    "xy_keys": ["text", "label"],  # text,label在file中对应的keys
    "label_sep": "|myz|",  # "|myz|" 多标签数据分割符, 用于多标签分类语料中
    "len_rate": 1,  # 训练数据和验证数据占比, float, 0-1闭区间
    "adv_emb_name": "word_embeddings.",  # emb_name这个参数要换成你模型中embedding的参数名, model.embeddings.word_embeddings.weight
    "adv_eps": 1.0,  # 梯度权重epsilon

    "ADDITIONAL_SPECIAL_TOKENS": ["[macropodus]", "[macadam]"],  # 新增特殊字符
    "len_corpus": None,  # 训练样本数, 自动更新
    "prior_count": None,  # 各个类别频次, 自动更新
    "prior": None,  # 类别先验分布, 自动更新, 为一个label_num类别数个元素的list, json无法保存np.array
    "l2i": None,
    "i2l": None,
}


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = model_config.get("CUDA_VISIBLE_DEVICES", "2")
os.environ["USE_TORCH"] = model_config.get("USE_TORCH", "1")
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer, ElectraTokenizer, XLMTokenizer, AutoTokenizer
from transformers import BertConfig, RobertaConfig, AlbertConfig, XLNetConfig, ElectraConfig, XLMConfig, AutoConfig
from transformers import BertModel, RobertaModel, AlbertModel, XLNetModel, ElectraModel, XLMModel, AutoModel
# from transformers import LongformerTokenizer, LongformerConfig, LongformerModel
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model
from transformers import T5Tokenizer, T5Config, T5Model


# transformers类等
PRETRAINED_MODEL_CLASSES = {
    # "LONGFORMER": (LongformerConfig, LongformerTokenizer, LongformerModel),
    "ELECTRA": (ElectraConfig, ElectraTokenizer, ElectraModel),
    "ROBERTA": (AutoConfig, AutoTokenizer, AutoModel),  # (RobertaConfig, RobertaTokenizer, RobertaModel),  #
    "ALBERT": (AlbertConfig, AlbertTokenizer, AlbertModel),
    "MACBERT": (AutoConfig, BertTokenizer, BertModel),
    "XLNET": (XLNetConfig, XLNetTokenizer, XLNetModel),
    "ERNIE": (BertConfig, BertTokenizer, BertModel),
    "NEZHA": (BertConfig, BertTokenizer, BertModel),
    "BERT": (BertConfig, BertTokenizer, BertModel),
    "GPT2": (GPT2Config, GPT2Tokenizer, GPT2Model),
    "AUTO": (AutoConfig, AutoTokenizer, AutoModel),
    "XLM": (XLMConfig, XLMTokenizer, XLMModel),
    "T5": (T5Config, T5Tokenizer, T5Model)
}

