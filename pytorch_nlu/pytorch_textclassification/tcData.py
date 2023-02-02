# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: deal corpus(data preprocess), label全部转成onehot的形式


import logging as logger
from abc import ABC
import json
import os

from torch.utils.data import TensorDataset
import torch

from tcConfig import PRETRAINED_MODEL_CLASSES
from tcTqdm import tqdm


class Corpus(ABC):
    def __init__(self, config, logger=logger):
        self.config = config
        self.logger = logger
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.tokenizer = self.load_tokenizer(self.config)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.unk_token = self.tokenizer.unk_token
        self.max_len = self.config.max_len
        self.l2i, self.i2l = {}, {}

    def read_corpus_from_json(self, path_json, encoding="utf-8", len_rate=1, keys=["text", "label"]):
        """
        从定制化的标准json文件中读取初始语料, read corpus from json
        config:
            path_json: str, path of corpus
            encoding: str, file encoding type, eg. "utf-8", "gbk"
            len_rate: float, 0-1, eg. 0.5
            keys: list, selected key of json
        Returns:
            (xs, ys): tuple
        """
        xs, ys, len_maxs = [], [], []
        count = 0
        with open(path_json, "r", encoding=encoding) as fo:
            for line in fo:
                count += 1
                # if count > 32:
                #     break
                if not line:
                    continue
                #  最初想定义可配置化, 但是后期实验较多, 还是设置成一般形式, 可自己定义
                line_json = json.loads(line.strip())
                x, y = line_json.get(keys[0], ""), line_json.get(keys[1], "")
                len_maxs.append(len(x))
                xs.append((x, y))
                ys.append(y)

            fo.close()
            xs.append((x, y))
            ys.append(y)
            # 没有验证集的情况
            len_rel = int(count * len_rate) if 0<len_rate<1 else count
            xs = xs[:len_rel+1]
            ys = ys[:len_rel+1]
            # 分析统计文本长度, 覆盖0.95的长度
            len_maxs.sort()
            len_max_100 = len_maxs[-1]
            len_max_95 = len_maxs[int(len(len_maxs) * 0.95)]
            len_max_90 = len_maxs[int(len(len_maxs) * 0.90)]
            len_max_50 = len_maxs[int(len(len_maxs) * 0.50)]
            self.logger.info("len_max_100: {}".format(len_max_100))
            self.logger.info("len_max_95: {}".format(len_max_95))
            self.logger.info("len_max_90: {}".format(len_max_90))
            self.logger.info("len_max_50: {}".format(len_max_50))
            if self.config.max_len == 0:  # 即ax_len为0则是强制获取语料中的最大文本长度
                self.max_len = min(max(len_maxs) + 2, 512)
            elif self.config.max_len is None or self.config.max_len == -1:
                self.max_len = min(len_max_95 + 2, 512)
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
            x, y = line_json.get(keys[0], ""), line_json.get(keys[1], "")
            xs.append((x, y))
            ys.append(y)
        return xs, ys

    def preprocess(self, data_iter, label2idx, max_len=512, label_sep="|xyz|"):
        """
        pre-process with x(sequence)
        config:
            data_iter: iter, iter of (x, y), eg. ("你是谁", "问句")
            label2idx: dict, dict of label to number, eg. {"问句":0}
            max_len: int, max length of text, eg. 512
            use_seconds: bool, either use [SEP] separate texts2 or not, eg.True
            is_multi: bool, either sign sentence in texts with multi or not, eg. True
            label_sep: str, sign of multi-label split, eg. "#", "|@|" 
        Returns:
            inputs of bert-like model
        """

        batch_attention_mask_ids = []
        batch_token_type_ids = []
        batch_input_ids = []
        batch_label_ids = []
        len_label = len(label2idx)
        count = 0
        qbar = tqdm(data_iter, desc="data_preprocess") if self.config.is_train==True else data_iter
        for di in qbar:
        # for di in data_iter:
            count += 1
            x, y = di
            tokens = self.tokenizer.tokenize(x)
            ### 超出长度则首尾截断
            if len(tokens) > max_len:   ### token
                mid_maxlen = int(max_len / 2) - 1
                tokens = tokens[:mid_maxlen] + tokens[-mid_maxlen:]

            token_type_ids = [0] * max_len
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # cls_id_1 = self.tokenizer.convert_tokens_to_ids(["[unused2]"])
            # cls_id_2 = self.tokenizer.convert_tokens_to_ids(["[unused3]"])
            #  padding
            len_more = 2  # 4
            pad_len = max_len - len(input_ids) - len_more
            if max_len - len(input_ids) - len_more >= 0:
                input_ids = [self.cls_token_id] + input_ids + [0]*pad_len + [self.sep_token_id]
                # input_ids = cls_id_1 + cls_id_2 + [self.cls_token_id] + input_ids + [0] * pad_len + [self.sep_token_id]
                attention_mask_ids = [1] * (max_len-pad_len-1) + [0] * (pad_len+1)
            else:
                input_ids = [self.cls_token_id] + input_ids[:max_len-len_more] + [self.sep_token_id]
                # input_ids = cls_id_1 + cls_id_2 + [self.cls_token_id] + input_ids[:max_len - len_more] + [self.sep_token_id]
                attention_mask_ids = [1] * max_len
            #  label 全部转为 onehot
            label_ids = [0] * len_label
            for lab in y.split(label_sep):
                if lab and lab in label2idx:
                    label_ids[label2idx[lab]] = 1
            # if 1 not in label_ids:
            #     ee = 0
            batch_attention_mask_ids.append(attention_mask_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_input_ids.append(input_ids)
            batch_label_ids.append(label_ids)
            if count <= 5 and self.config.is_train:
                self.logger.info("****** Sample ******")
                self.logger.info("token: %s", " ".join([str(x) for x in tokens]))
                self.logger.info("input_id: %s", " ".join([str(x) for x in input_ids]))
                self.logger.info("token_type_id: %s", " ".join([str(x) for x in token_type_ids]))
                self.logger.info("attention_mask_id: %s", " ".join([str(x) for x in attention_mask_ids]))
                self.logger.info("label_id: %s" % " ".join([str(x) for x in label_ids]))
        # tensor
        tensor_attention_mask_ids = torch.tensor(batch_attention_mask_ids, dtype=torch.long)
        tensor_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        tensor_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        tensor_label_ids = torch.tensor(batch_label_ids, dtype=torch.float32)
        tensor_data = TensorDataset(tensor_input_ids, tensor_attention_mask_ids, tensor_token_type_ids, tensor_label_ids)
        return tensor_data

    def load_tokenizer(self, config):
        """
        加载标记器, load tokenizer
        config:
            config: dict, enum of parms
        Returns:
            tokenizer: class
        """
        class PretrainTokenizer(PRETRAINED_MODEL_CLASSES[config.model_type][1]):
            """ 强制单个字token, 避免自带的tokenize删除空白、或者是其他特殊字符的情况 """
            def tokenize(self, text):
                tokens = []
                for t in text:
                    if self.do_lower_case:
                        t = t.lower()
                    if t in self.vocab:
                        tokens.append(t)
                    # elif not t.replace(" ", "").strip():
                    #     tokens.append("[unused1]")
                    else:
                        tokens.append(self.unk_token)
                return tokens
        if config.tokenizer_type.upper() == "BASE":
            tokenizer = PRETRAINED_MODEL_CLASSES[config.model_type][1].from_pretrained(config.pretrained_model_name_or_path)
        else:
            tokenizer = PretrainTokenizer.from_pretrained(config.pretrained_model_name_or_path)  # 改写了以后会报错
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        for ast in self.ADDITIONAL_SPECIAL_TOKENS:
            if ast not in tokenizer.vocab:
                tokenizer.vocab[ast] = len(tokenizer.vocab)
        return tokenizer

