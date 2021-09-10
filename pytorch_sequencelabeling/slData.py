# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: deal corpus(data preprocess), label转成list形式, 或者span的形式


from slConfig import _SL_MODEL_SOFTMAX, _SL_MODEL_GRID, _SL_MODEL_SPAN, _SL_MODEL_CRF
from slConfig import _SL_DATA_CONLL, _SL_DATA_SPAN
from slConfig import PRETRAINED_MODEL_CLASSES
from slTools import transform_span_to_conll
from slTools import get_pos_from_common
from torch.utils.data import TensorDataset
import torch
import logging as logger
from abc import ABC
import json


class Corpus(ABC):
    def __init__(self, config, logger=logger):
        self.config = config
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.tokenizer = self.load_tokenizer(self.config)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.l2i, self.i2l = {}, {}
        self.logger = logger
        self.len_max = self.config.max_len

    def read_corpus_from_conll(self, path, keys=[0,1], row_sep=" ", encoding="utf-8"):
        """
        从定制化的标准json文件中读取初始语料, read corpus from json
        args:
            path: str, path of ner-corpus
            encoding: str, file encoding type, eg. "utf-8", "gbk"
            keys: list<int>, selected key-index of json, eg. [0,1], [1,3] 
            row_sep: str, str sep of split, eg. " ", "\t"
        returns:
            (xs, ys): tuple
        examples:
            中 B-CONT
            国 M-CONT
            国 M-CONT
            籍 E-CONT
            ， O
        """
        with open(path, "r", encoding=encoding) as fo:
            xs, ys, len_maxs = [], [], []
            x, y = [], []
            count = 0
            for line in fo:
                count += 1
                # if count > 32:
                #     break
                line_sp = line.strip().split(row_sep)
                if len(line_sp) == 1:
                    xs.append(("".join(x), y))
                    ys.append(y)
                    x, y = [], []
                else:
                    x.append(line_sp[keys[0]])
                    y.append(line_sp[keys[1]])
                len_maxs.append(len(x))
            fo.close()
            # 如果自动则选择覆盖0.95的长度, 即max_len为None, -1的情形
            len_maxs.sort()
            len_max_100 = len_maxs[-1]
            len_max_95 = len_maxs[int(len(len_maxs) * 0.95)]
            len_max_90 = len_maxs[int(len(len_maxs) * 0.90)]
            len_max_50 = len_maxs[int(len(len_maxs) * 0.50)]
            self.logger.info("len_max_100: {}".format(len_max_100))
            self.logger.info("len_max_95: {}".format(len_max_95))
            self.logger.info("len_max_90: {}".format(len_max_90))
            self.logger.info("len_max_50: {}".format(len_max_50))
            if not self.config.max_len or self.config.max_len == -1:
                self.len_max = len_max_95
            elif self.config.max_len == 0:  # 即ax_len为0则是强制获取语料中的最大文本长度
                self.len_max = max(len_maxs) + 2
            return xs, ys

    def read_corpus_from_span(self, path, encoding="utf-8", keys=["text", "label"]):
        """
        从定制化的标准json文件中读取初始语料, read corpus from myz
        config:
            path_json: str, path of corpus
            encoding: str, file encoding type, eg. "utf-8", "gbk"
            keys: list, selected key of json
        Returns:
            (xs, ys): tuple
        """
        with open(path, "r", encoding=encoding) as fo:
            xs, ys, len_maxs = [], [], []
            count = 0
            for line in fo:
                count += 1
                # if count > 32:
                #     break
                if not line:
                    continue
                #  最初想定义可配置化, 但是后期实验较多, 还是设置成一般形式, 可自己定义
                line_json = json.loads(line.strip())
                x, y = line_json.get(keys[0], ""), line_json.get(keys[1], [])
                len_maxs.append(len(x))
                xs.append((x, y))
                ys.append(y)
            fo.close()
            # 如果自动则选择覆盖0.95的长度, 即max_len为None, -1的情形
            len_maxs.sort()
            len_max_100 = len_maxs[-1]
            len_max_95 = len_maxs[int(len(len_maxs) * 0.95)]
            len_max_90 = len_maxs[int(len(len_maxs) * 0.90)]
            len_max_50 = len_maxs[int(len(len_maxs) * 0.50)]
            self.logger.info("len_max_100: {}".format(len_max_100))
            self.logger.info("len_max_95: {}".format(len_max_95))
            self.logger.info("len_max_90: {}".format(len_max_90))
            self.logger.info("len_max_50: {}".format(len_max_50))
            if not self.config.max_len or self.config.max_len == -1:
                self.len_max = len_max_95
            elif self.config.max_len == 0:  # 即max_len为0则是强制获取语料中的最大文本长度
                self.len_max = max(len_maxs) + 2
            return xs, ys

    def read_texts_from_json(self, texts, keys=["text", "label"]):
        """
        一般预测用, 从列表texts中获取json, read corpus from texts
        config:
            texts: List<json>, eg. [{"text":"12306", "label":"yes"}]
            encoding: str, file encoding type, eg. "utf-8", "gbk"
            keys: list, selected key of json, eg. ["text", "label"]
        Returns:
            (xs, ys): tuple
        """
        xs, ys = [], []
        count = 0
        for line_json in texts:
            count += 1
            # if count > 32:
            #     break
            if not line_json:
                continue
            #  最初想定义可配置化, 但是后期实验较多, 还是设置成一般形式, 可自己定义
            x, y = line_json.get(keys[0], ""), line_json.get(keys[1], [])
            xs.append((x, y))
            ys.append(y)
        return xs, ys

    def preprocess_common(self, data_iter, label2idx, max_len=512, sl_ctype="BIO", l2i_conll=None):
        """  sequence-labeling, 序列标注任务
        pre-process with x(sequence)
        config:
            data_iter: iter, iter of (x, y), eg. ("你是谁", "问句")
            label2idx: dict, dict of label to number, eg. {"问句":0}
            max_len: int, max length of text, eg. 512
            use_seconds: bool, either use [SEP] separate texts2 or not, eg.True
            is_multi: bool, either sign sentence in texts with multi or not, eg. True
            label_sep: str, sign of multi-label split, eg. "#", "|@|" 
            sl_ctype: str, corpus-data-type of sequence-labeling, eg.BIO, IOB, BMES, BIOES
        Returns:
            inputs of bert-like model
        """
        batch_attention_mask = []
        batch_token_type = []
        batch_input = []
        batch_label = []
        batch_text = []
        count = 0
        for di in data_iter:
            count += 1
            x, y = di
            token = self.tokenizer.tokenize(x)
            token_type_id = [0] * max_len
            input_id = self.tokenizer.convert_tokens_to_ids(token)
            # padding到最大文本长度
            pad_len = max_len - len(input_id) - 2
            if max_len - len(input_id) - 2 >= 0:
                input_id = [self.cls_token_id] + input_id + [0] * pad_len + [self.sep_token_id]
                attention_mask_id = [1] * (max_len - pad_len - 1) + [0] * (pad_len + 1)
            else:
                input_id = [self.cls_token_id] + input_id[:max_len - 2] + [self.sep_token_id]
                attention_mask_id = [1] * max_len
            # ner-label 全部转为 onehot, 0是最多的O
            label_id = [0] * len(input_id)
            if self.config.corpus_type == _SL_DATA_CONLL:  # conll格式, 如已经存在的BMES、BIO等, eg. {"text": "南京", "label":["B-city", "I-city"]}
                for i, yi in enumerate(y):
                    if yi in label2idx and i < len(input_id)-1:
                        label_id[i+1] = label2idx[yi]
            elif self.config.corpus_type == _SL_DATA_SPAN:  # myx格式, 嵌套为json, eg. {"text": "南京", "label":[{"ent":"南京", "type": "city", "pos":[0,1]}]}
                # 数据格式, sl-type, BIO, BMES, BIOES
                if y:
                    label_id = transform_span_to_conll(y, label_id, l2i_conll, sl_ctype)

            batch_attention_mask.append(attention_mask_id)
            batch_token_type.append(token_type_id)
            batch_input.append(input_id)
            batch_label.append(label_id)
            batch_text.append(x)
            # logger
            if count <= 5 and self.config.is_train:
                self.logger.info("*** Sample ***")
                self.logger.info("text: %s", x)
                self.logger.info("token: %s", " ".join([str(x) for x in token]))
                self.logger.info("input_id: %s", " ".join([str(x) for x in input_id]))
                self.logger.info("token_type_id: %s", " ".join([str(x) for x in token_type_id]))
                self.logger.info("attention_mask_id: %s", " ".join([str(x) for x in attention_mask_id]))
                self.logger.info("label_id: %s" % " ".join([str(x) for x in label_id]))
        # tensor
        tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        tensor_token_type = torch.tensor(batch_token_type, dtype=torch.long)
        tensor_label = torch.tensor(batch_label, dtype=torch.float32)
        tensor_input = torch.tensor(batch_input, dtype=torch.long)
        tensor_data = TensorDataset(tensor_input, tensor_attention_mask, tensor_token_type, tensor_label)
        return tensor_data, batch_text

    def preprocess_span(self, data_iter, label2idx, max_len=512, sl_ctype="BIO", l2i_conll=None):
        """  sequence-labeling, 序列标注任务
        pre-process with x(sequence)
        config:
            data_iter: iter, iter of (x, y), eg. ("你是谁", [{"ent":"北京", "type":"LOC", "pos":[0,1]}])
            label2idx: dict, dict of label to number, eg. {"问句":0}
            max_len: int, max length of text, eg. 512
            use_seconds: bool, either use [SEP] separate texts2 or not, eg.True
            is_multi: bool, either sign sentence in texts with multi or not, eg. True
            label_sep: str, sign of multi-label split, eg. "#", "|@|" 
        Returns:
            inputs of bert-like model
        """
        batch_attention_mask = []
        batch_token_type = []
        batch_input = []
        batch_start = []
        batch_end = []
        batch_text = []
        count = 0
        for di in data_iter:
            count += 1
            x, y = di
            token = self.tokenizer.tokenize(x)
            token_type_id = [0] * max_len
            input_id = self.tokenizer.convert_tokens_to_ids(token)
            # padding到最大文本长度
            pad_len = max_len - len(input_id) - 2
            if max_len - len(input_id) - 2 >= 0:
                input_id = [self.cls_token_id] + input_id + [0] * pad_len + [self.sep_token_id]
                attention_mask_id = [1] * (max_len - pad_len - 1) + [0] * (pad_len + 1)
            else:
                input_id = [self.cls_token_id] + input_id[:max_len - 2] + [self.sep_token_id]
                attention_mask_id = [1] * max_len
            # ner-label 全部转为 onehot
            start_id = [0] * len(input_id)
            end_id = [0] * len(input_id)
            if self.config.corpus_type == _SL_DATA_CONLL:  # conll格式, eg. {"text": "南京", "label":["B-city", "I-city"]}
                for i, yi in enumerate(y):
                    sep = "-"
                    yi_sp = yi.split(sep)
                    yi_type = yi_sp[-1]
                    if yi_type in label2idx and i < len(input_id)-1:  # 存成span格式, 即开头和结尾 / 有-的类别, 如S-city, B-LOC
                        if i==0 or (i>0 and y[i-1].split(sep)[-1] != yi_type):  # 开始时候, 或者前一个类别不等于当前的情况, 记一start
                            start_id[i+1] = label2idx[yi_type]
                        if i==len(y)-1 or (i < len(y)-1 and y[i+1].split(sep)[-1] != yi_type):  # 结尾时候, 或者后一个类别不等于当前的情况, 记一end
                            end_id[i+1] = label2idx[yi_type]
            elif self.config.corpus_type == _SL_DATA_SPAN:  # myx格式, 嵌套为json, eg. {"text": "南京", "label":[{"ent":"南京", "type": "city", "pos":[0,1]}]}
                for i, yi in enumerate(y):
                    yi_pos = yi.get("pos", [0, 1])
                    yi_type = yi.get("type", "")
                    # yi_e = yi.get("ent", "")
                    if yi_type in label2idx and yi_pos[1] < len(input_id)-1:
                        start_id[yi_pos[0]+1] = label2idx[yi_type]
                        end_id[yi_pos[1]+1] = label2idx[yi_type]
            batch_attention_mask.append(attention_mask_id)
            batch_token_type.append(token_type_id)
            batch_input.append(input_id)
            batch_start.append(start_id)
            batch_end.append(end_id)
            batch_text.append(x)
            # logger
            if count <= 5 and self.config.is_train:
                self.logger.info("*** Sample ***")
                self.logger.info("text: %s", x)
                self.logger.info("token: %s", " ".join([str(x) for x in token]))
                self.logger.info("input_id: %s", " ".join([str(x) for x in input_id]))
                self.logger.info("token_type_id: %s", " ".join([str(x) for x in token_type_id]))
                self.logger.info("attention_mask_id: %s", " ".join([str(x) for x in attention_mask_id]))
                self.logger.info("start_id: %s" % " ".join([str(x) for x in start_id]))
                self.logger.info("end_id: %s" % " ".join([str(x) for x in end_id]))
        # tensor
        tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        tensor_token_type = torch.tensor(batch_token_type, dtype=torch.long)
        tensor_input = torch.tensor(batch_input, dtype=torch.long)
        tensor_start = torch.tensor(batch_start, dtype=torch.float32)
        tensor_end = torch.tensor(batch_end, dtype=torch.float32)
        tensor_data = TensorDataset(tensor_input, tensor_attention_mask, tensor_token_type, tensor_start, tensor_end)
        return tensor_data, batch_text

    def preprocess_grid(self, data_iter, label2idx, max_len=512, sl_ctype="BIO", l2i_conll=None):
        """  sequence-labeling, 序列标注任务
        pre-process with x(sequence)
        config:
            data_iter: iter, iter of (x, y), eg. ("你是谁", [{"ent":"北京", "type":"LOC", "pos":[0,1]}])
            label2idx: dict, dict of label to number, eg. {"问句":0}
            max_len: int, max length of text, eg. 512
            use_seconds: bool, either use [SEP] separate texts2 or not, eg.True
            is_multi: bool, either sign sentence in texts with multi or not, eg. True
            label_sep: str, sign of multi-label split, eg. "#", "|@|" 
        Returns:
            inputs of bert-like model
        """
        batch_attention_mask = []
        batch_token_type = []
        batch_input = []
        batch_grid = []
        batch_text = []
        count = 0
        for di in data_iter:
            count += 1
            x, y = di
            token = self.tokenizer.tokenize(x)
            token_type_id = [0] * max_len
            input_id = self.tokenizer.convert_tokens_to_ids(token)  # 全部当成中文处理, 杜绝 "雷吉纳vsac米兰" 问题
            # padding到最大文本长度
            pad_len = max_len - len(input_id) - 2
            if max_len - len(input_id) - 2 >= 0:
                input_id = [self.cls_token_id] + input_id + [0] * pad_len + [self.sep_token_id]
                attention_mask_id = [1] * (max_len - pad_len - 1) + [0] * (pad_len + 1)
            else:
                input_id = [self.cls_token_id] + input_id[:max_len - 2] + [self.sep_token_id]
                attention_mask_id = [1] * max_len
            # ner-label 全部转为 onehot
            grid = [[[0 for _ in range(max_len)] for _ in range(max_len)] for _ in range(len(label2idx))]
            grid_span = None
            if self.config.corpus_type == _SL_DATA_CONLL:  # conll格式, eg. {"text": "南京", "label":["B-city", "I-city"]}
                y_conll = get_pos_from_common(x, y, sep="-")  # 支持BIO, BIOES, BMES
                grid_span = y_conll
                for i, yi in enumerate(y_conll):
                    yi_pos = yi.get("pos", [0, 1])
                    yi_type = yi.get("type", "")
                    if yi_type in label2idx and yi_pos[1] < len(input_id) - 1:
                        grid[label2idx[yi_type]][yi_pos[0] + 1][yi_pos[1] + 1] = 1
            elif self.config.corpus_type == _SL_DATA_SPAN:  # myx格式, 嵌套为json, eg. {"text": "南京", "label":[{"ent":"南京", "type": "city", "pos":[0,1]}]}
                grid_span = y
                for i, yi in enumerate(y):
                    yi_pos = yi.get("pos", [0, 1])
                    yi_type = yi.get("type", "")
                    # yi_e = yi.get("ent", "")
                    if yi_type in label2idx and yi_pos[1] < len(input_id)-1:
                        grid[label2idx[yi_type]][yi_pos[0]+1][yi_pos[1]+1] = 1
            batch_attention_mask.append(attention_mask_id)
            batch_token_type.append(token_type_id)
            batch_input.append(input_id)
            batch_grid.append(grid)
            batch_text.append(x)
            # logger
            if count <= 5 and self.config.is_train:
                self.logger.info("*** Sample ***")
                self.logger.info("text: %s", x)
                self.logger.info("token: %s", " ".join([str(x) for x in token]))
                self.logger.info("input_id: %s", " ".join([str(x) for x in input_id]))
                self.logger.info("token_type_id: %s", " ".join([str(x) for x in token_type_id]))
                self.logger.info("attention_mask_id: %s", " ".join([str(x) for x in attention_mask_id]))
                self.logger.info("grid_span: {}".format(grid_span[:5]))
        # tensor
        tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        tensor_token_type = torch.tensor(batch_token_type, dtype=torch.long)
        tensor_input = torch.tensor(batch_input, dtype=torch.long)
        tensor_grid = torch.tensor(batch_grid, dtype=torch.float32)
        tensor_data = TensorDataset(tensor_input, tensor_attention_mask, tensor_token_type, tensor_grid)
        return tensor_data, batch_text

    def load_tokenizer(self, config):
        """
        加载标记器, load tokenizer
        config:
            config: dict, enum of parms
        Returns:
            tokenizer: class
        """
        class PretrainedTokenizer(PRETRAINED_MODEL_CLASSES[config.model_type][1]):
            """ 避免自带的tokenize删除空白、或者是其他特殊字符的情况 """
            def tokenize(self, text):
                tokens = []
                for t in text:
                    if self.do_lower_case:
                        t = t.lower()
                    if t in self.vocab:
                        tokens.append(t)
                    else:
                        tokens.append("[UNK]")
                return tokens

        tokenizer = PretrainedTokenizer.from_pretrained(config.pretrained_model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer

