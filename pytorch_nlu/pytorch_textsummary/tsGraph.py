# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:35
# @author  : Mo
# @function: graph of pre-train model


# torch
from transformers import BertPreTrainedModel
import torch

from tsConfig import PRETRAINED_MODEL_CLASSES


class TSGraph(BertPreTrainedModel):
    def __init__(self, graph_config, tokenizer):
        """
        Pytorch Graph of TextSummary, Pre-Trained Model based
        config:
            config: json, params of graph, eg. {"num_labels":17, "model_type":"BERT"}
        Returns:
            output: Tuple, Tensor of logits and loss
        Url: https://github.com/yongzhuo
        """
        # 预训练语言模型读取
        self.graph_config = graph_config
        pretrained_config, pretrained_tokenizer, pretrained_model = PRETRAINED_MODEL_CLASSES[graph_config.model_type]
        self.pretrained_config = pretrained_config.from_pretrained(graph_config.pretrained_model_name_or_path, output_hidden_states=graph_config.output_hidden_states)
        self.pretrained_config.update({"gradient_checkpointing": True})
        # self.tokenizer = pretrained_tokenizer.from_pretrained(graph_config.pretrained_model_name_or_path)
        # self.tokenizer = tokenizer
        super(TSGraph, self).__init__(self.pretrained_config)
        if self.graph_config.is_train:
            self.pretrain_model = pretrained_model.from_pretrained(graph_config.pretrained_model_name_or_path, config=self.pretrained_config)
            self.pretrain_model.resize_token_embeddings(len(tokenizer))
        else:
            self.pretrain_model = pretrained_model(self.pretrained_config)
            self.pretrain_model.resize_token_embeddings(len(tokenizer))
        # tokenizer.model_max_length = self.model.config.max_position_embeddings
        # 如果用隐藏层输出
        self.dense = torch.nn.Linear(self.pretrained_config.hidden_size, 1)

        # 池化层
        self.global_maxpooling = torch.nn.AdaptiveMaxPool1d(1)
        self.global_avgpooling = torch.nn.AdaptiveAvgPool1d(1)

        # 激活层/随即失活层
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout()

    def forward(self, input_ids, attention_mask, token_type_ids, mask_cls, cls_ids, labels=None):
        output = self.pretrain_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        top_vec = output.last_hidden_state
        if self.graph_config.is_dropout:
            top_vec = self.dropout(top_vec)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), cls_ids]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        h = self.dense(sents_vec).squeeze(-1)
        # sent_scores = torch.nn.Sigmoid()(h) * mask_cls.float()
        sent_scores = h * mask_cls.float()
        logits = sent_scores.squeeze(-1)
        # inference
        if self.graph_config.is_fc_sigmoid:
            return self.sigmoid(logits)
        elif self.graph_config.is_fc_softmax:
            return self.softmax(logits)
        return logits

