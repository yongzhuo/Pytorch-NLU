# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: deal corpus(data preprocess), label全部转成onehot的形式


from collections import Counter
import logging as logger
from abc import ABC
import random
import json

from tqdm import tqdm
import torch

from tsConfig import PRETRAINED_MODEL_CLASSES


class DataSet(ABC):
    def __init__(self, config, path_json=None, logger=logger):
        self.config = config
        self.logger = logger
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.tokenizer = self.load_tokenizer(self.config)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.l2i, self.i2l = {}, {}
        self.max_len = self.config.max_len
        self.label_sep = self.config.label_sep
        self.data_iter = None
        self.len_corpus = 0
        if path_json:
            self.read_corpus_from_json(path_json, keys=self.config.xy_keys, len_rate=self.config.len_rate)

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
        xs, len_maxs = [], []
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
            fo.close()

        random.shuffle(xs)
        ys = [ysi[-1] for ysi in xs]
        len_rel = int(count * len_rate) if 0<len_rate<1 else count
        xs = xs[:len_rel+1]
        ys = ys[:len_rel+1]

        # 覆盖0.95的长度
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
            self.len_max = min(max(len_maxs) + 2, 512)
        elif self.config.max_len is None or self.config.max_len == -1:
            self.len_max = min(len_max_95 + 2, 512)

        counter_dict = {}
        for yt in ys:
            counter_yt_dict = dict(Counter(yt))
            for k, v in counter_yt_dict.items():
                if k not in counter_dict:
                    counter_dict[str(k)] = v
                else:
                    counter_dict[str(k)] += v
        prior_count = [counter_dict["0"], counter_dict["1"]]
        prior = [pc / sum(counter_dict.values()) for pc in prior_count]
        self.logger.info("prior-label: {}".format(prior))
        self.prior_count = prior_count
        self.prior = prior
        self.data_iter = xs, ys
        self.len_corpus = len(ys)

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

    def sequence_padding(self, inputs, length=None, padding=0):
        """
        Numpy函数，将序列padding到同一长度
        config:
            config: dict, enum of parms
        Returns:
            tokenizer: class
        """
        if length is None:
            length = min(max([len(x) for x in inputs]), 512)
        outputs = []
        for x in inputs:
            if len(x) >= length:
                x = x[:length]
            else:
                x = x + [padding] * (length-len(x))
            outputs.append(x)
        return outputs

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
            tokenizer = PretrainTokenizer.from_pretrained(config.pretrained_model_name_or_path)  # 改写了以后可能会报错
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        for ast in self.ADDITIONAL_SPECIAL_TOKENS:
            if ast not in tokenizer.vocab:
                tokenizer.vocab[ast] = len(tokenizer.vocab)
        return tokenizer

    def convert_text_to_ids(self, xs_mid):
        """
        文本text转数值id
        config:
            xs_mid<List<Str>>: input of texts, eg. ["123", "456"]
        Returns:
            attention_mask_ids, token_type_ids, input_ids, cls_ids
        """
        # 单页中间截断
        input_tokens = []
        input_ids = []
        for x in xs_mid:
            tokens = self.tokenizer.tokenize(x)
            tokens_id = [self.cls_token_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.sep_token_id]
            input_ids.extend(tokens_id)
            input_tokens.extend([self.cls_token] + tokens + [self.sep_token])
        _token_type = [-1] + [i for i, t in enumerate(input_ids) if t == self.sep_token_id]
        _token_type = [_token_type[i] - _token_type[i - 1] for i in range(1, len(_token_type))]
        token_type_ids = []
        for i, tt in enumerate(_token_type):  # 多句子输入, 类似1111100000111111
            if (i % 2 == 0):
                token_type_ids.extend(tt * [0])
            else:
                token_type_ids.extend(tt * [1])
        attention_mask_ids = [1] * len(input_ids)
        cls_ids = [i for i, t in enumerate(input_ids) if t == self.cls_token_id]
        return attention_mask_ids, token_type_ids, input_ids, cls_ids, input_tokens

    def __pred__(self, desc="iter"):
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
        batch_cls_ids = []
        batch_token_text = []
        self.len_corpus = len(self.data_iter[1])
        for idx in range(self.len_corpus):
            xs_ys = self.data_iter[0][idx]
            xs, ys = xs_ys
            len_ys = len(xs)
            len_x_max = 0
            jdx_start = 0
            # 拼接/截断, 统计文本长度, start/end
            for jdx, x in enumerate(xs):
                len_x_max += len(x)
                ### 大于最大长度/结束, 开始拼接
                xs_mid = None
                flag_end = False  # 用于是判断最后一行, 且加上前面的文本大于maxlen的情况
                if len_ys - 1 == jdx:  # 结尾
                    if len_x_max + (jdx - jdx_start) * 2 >= self.max_len:  # sent1-sent2-sent3
                        xs_mid = xs[jdx_start:jdx]
                        flag_end = True
                    else:
                        len_curr = len("".join(xs[jdx_start:jdx])) + (jdx + 1 - jdx_start + 1) * 2
                        xs_mid = xs[jdx_start:jdx] + [xs[jdx][:(self.max_len - len_curr)]]
                elif len_x_max + (jdx-jdx_start) * 2 >= self.max_len:  # sent1-sent2-sent3
                    xs_mid = xs[jdx_start:jdx]

                ### 上边两种情况
                if xs_mid:
                    # 从中间页面截断
                    jdx_start = jdx
                    len_x_max = len(x)
                    # ***注意***, 当前页需要计算
                    attention_mask_ids, token_type_ids, input_ids, cls_ids, input_tokens = self.convert_text_to_ids(xs_mid)
                    batch_attention_mask_ids.append(attention_mask_ids)
                    batch_token_type_ids.append(token_type_ids)
                    batch_input_ids.append(input_ids)
                    batch_cls_ids.append(cls_ids)
                    batch_token_text.append(xs_mid)
                    if flag_end:  # 用于是判断最后一行, 且加上前面的文本大于maxlen的情况
                        xs_mid = [xs[jdx]]
                        attention_mask_ids, token_type_ids, input_ids, cls_ids, input_tokens = self.convert_text_to_ids(xs_mid)
                        batch_attention_mask_ids.append(attention_mask_ids)
                        batch_token_type_ids.append(token_type_ids)
                        batch_input_ids.append(input_ids)
                        batch_cls_ids.append(cls_ids)
                        batch_token_text.append(xs_mid)

                    # batch-size 或 结束
                    # if len(batch_label_ids) == self.config.batch_size or self.len_corpus-1 == idx:
                    if len(batch_label_ids) >= self.config.batch_size or len_ys - 1 == jdx:
                        batch_attention_mask_ids = self.sequence_padding(batch_attention_mask_ids, length=None, padding=0)
                        batch_token_type_ids = self.sequence_padding(batch_token_type_ids, length=None, padding=0)
                        batch_input_ids = self.sequence_padding(batch_input_ids, length=None, padding=0)
                        batch_cls_ids = self.sequence_padding(batch_cls_ids, length=None, padding=-1)
                        # batch_mask_ids = ~(batch_input_ids == 0)
                        # batch_mask_cls_ids = [~(b == -1) for b in batch_cls_ids]
                        # batch_cls_ids[batch_cls_ids == -1] = 0

                        tensor_attention_mask_ids = torch.tensor(batch_attention_mask_ids, dtype=torch.long)
                        tensor_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
                        tensor_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
                        tensor_cls_ids = torch.tensor(batch_cls_ids, dtype=torch.long)
                        tensor_mask_cls_ids = ~(tensor_cls_ids == -1)  ### 非mask
                        tensor_cls_ids[tensor_cls_ids == -1] = 0
                        # list_mask_cls_ids = tensor_mask_cls_ids.detach().cpu().numpy()
                        # tensor_mask_ids = torch.tensor(batch_mask_ids, dtype=torch.long)
                        # tensor_mask_cls_ids = torch.tensor(batch_mask_cls_ids, dtype=torch.long)
                        yield tensor_attention_mask_ids, tensor_token_type_ids, tensor_input_ids, \
                              tensor_cls_ids, tensor_mask_cls_ids, batch_token_text
                        batch_attention_mask_ids = []
                        batch_token_type_ids = []
                        batch_input_ids = []
                        batch_label_ids = []
                        batch_cls_ids = []
                        batch_token_text = []

    def __iter__(self, desc="iter"):
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
        batch_cls_ids = []
        self.len_corpus = len(self.data_iter[1])
        for idx in tqdm(range(self.len_corpus), desc=desc):
            xs_ys = self.data_iter[0][idx]
            xs, ys = xs_ys
            len_ys = len(xs)
            len_x_max = 0
            jdx_start = 0
            for jdx, x in enumerate(xs):
                len_x_max += len(x)
                ### 大于最大长度/结束, 开始拼接
                xs_mid = None
                flag_end = False  # 用于是判断最后一行, 且加上前面的文本大于maxlen的情况
                if len_ys - 1 == jdx:  # 结尾
                    if len_x_max + (jdx - jdx_start) * 2 >= self.max_len:  # 最后一行不加到中间的情况
                        label_ids = ys[jdx_start:jdx]
                        xs_mid = xs[jdx_start:jdx]
                        flag_end = True
                    else:  # 最后一行加到前面的情况
                        len_curr = len("".join(xs[jdx_start:jdx])) + (jdx + 1 - jdx_start + 1) * 2
                        xs_mid = xs[jdx_start:jdx] + [xs[jdx][:(self.max_len - len_curr)]]
                        label_ids = ys[jdx_start:jdx + 1]
                elif len_x_max + (jdx - jdx_start) * 2 >= self.max_len:  # 中间截断, 如果加上jdx行大于maxlen就停止
                    xs_mid = xs[jdx_start:jdx]
                    label_ids = ys[jdx_start:jdx]
                ### 上边两种情况
                if xs_mid:
                    # 单页中间截断
                    jdx_start = jdx
                    len_x_max = len(x)
                    # ***注意***, 当前页需要计算
                    attention_mask_ids, token_type_ids, input_ids, cls_ids, input_tokens = self.convert_text_to_ids(xs_mid)
                    batch_attention_mask_ids.append(attention_mask_ids)
                    batch_token_type_ids.append(token_type_ids)
                    batch_input_ids.append(input_ids)
                    batch_label_ids.append(label_ids)
                    batch_cls_ids.append(cls_ids)

                    if flag_end:  # 最后一行不能加到前面的情况(即加上最后一行大于maxlen, 相当于最后一行单独成一个了)
                        xs_mid = [xs[jdx]]
                        label_ids = [ys[jdx]]
                        attention_mask_ids, token_type_ids, input_ids, cls_ids, input_tokens = self.convert_text_to_ids(xs_mid)
                        batch_attention_mask_ids.append(attention_mask_ids)
                        batch_token_type_ids.append(token_type_ids)
                        batch_input_ids.append(input_ids)
                        batch_label_ids.append(label_ids)
                        batch_cls_ids.append(cls_ids)

                    if idx <= 5 and self.config.is_train and len(batch_label_ids) < 5 and desc in ["train"]:
                        self.logger.info("****** Sample ******")
                        self.logger.info("token: %s", " ".join([str(x) for x in input_tokens]))
                        self.logger.info("input_id: %s", " ".join([str(x) for x in input_ids]))
                        self.logger.info("token_type_id: %s", " ".join([str(x) for x in token_type_ids]))
                        self.logger.info("attention_mask_id: %s", " ".join([str(x) for x in attention_mask_ids]))
                        self.logger.info("label_id: %s" % " ".join([str(x) for x in label_ids]))
                        self.logger.info("cls_id: %s" % " ".join([str(x) for x in cls_ids]))

                    # batch-size 或 结束
                    if len(batch_label_ids) >= self.config.batch_size or len_ys - 1 == jdx:
                        batch_attention_mask_ids = self.sequence_padding(batch_attention_mask_ids, length=None, padding=0)
                        batch_token_type_ids = self.sequence_padding(batch_token_type_ids, length=None, padding=0)
                        batch_input_ids = self.sequence_padding(batch_input_ids, length=None, padding=0)
                        batch_label_ids = self.sequence_padding(batch_label_ids, length=None, padding=0)
                        batch_cls_ids = self.sequence_padding(batch_cls_ids, length=None, padding=-1)
                        # batch_mask_ids = ~(batch_input_ids == 0)
                        # batch_mask_cls_ids = [~(b == -1) for b in batch_cls_ids]
                        # batch_cls_ids[batch_cls_ids == -1] = 0

                        tensor_attention_mask_ids = torch.tensor(batch_attention_mask_ids, dtype=torch.long)
                        tensor_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
                        tensor_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
                        tensor_label_ids = torch.tensor(batch_label_ids, dtype=torch.float32)
                        tensor_cls_ids = torch.tensor(batch_cls_ids, dtype=torch.long)
                        tensor_mask_cls_ids = ~(tensor_cls_ids == -1)  ### 非mask
                        tensor_cls_ids[tensor_cls_ids == -1] = 0
                        # list_mask_cls_ids = tensor_mask_cls_ids.detach().cpu().numpy()
                        # tensor_mask_ids = torch.tensor(batch_mask_ids, dtype=torch.long)
                        # tensor_mask_cls_ids = torch.tensor(batch_mask_cls_ids, dtype=torch.long)
                        yield tensor_attention_mask_ids, tensor_token_type_ids, tensor_input_ids, \
                              tensor_label_ids, tensor_cls_ids, tensor_mask_cls_ids
                        batch_attention_mask_ids = []
                        batch_token_type_ids = []
                        batch_input_ids = []
                        batch_label_ids = []
                        batch_cls_ids = []

    def __len__(self):
        """  获取corpus的文本长度  """
        return self.len_corpus

    def forfit(self, desc="iter"):
        while True:
            for it in self.__iter__(desc):
                yield it
