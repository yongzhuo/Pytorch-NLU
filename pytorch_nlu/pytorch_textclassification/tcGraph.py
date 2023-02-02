# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:35
# @author  : Mo
# @function: graph of pre-train model


from tcConfig import PRETRAINED_MODEL_CLASSES
from tcLayer import FCLayer
# torch
from transformers import BertPreTrainedModel
import torch


class TCGraph(BertPreTrainedModel):
    def __init__(self, graph_config, tokenizer):
        """
        Pytorch Graph of TextClassification, Pre-Trained Model based
        config:
            config: json, params of graph, eg. {"num_labels":17, "model_type":"BERT"}
        Returns:
            output: Tuple, Tensor of logits and loss
        """
        # 预训练语言模型读取
        self.graph_config = graph_config
        pretrained_config, pretrained_tokenizer, pretrained_model = PRETRAINED_MODEL_CLASSES[graph_config.model_type]
        self.pretrained_config = pretrained_config.from_pretrained(graph_config.pretrained_model_name_or_path, output_hidden_states=graph_config.output_hidden_states)
        self.pretrained_config.update({"gradient_checkpointing": True})
        super(TCGraph, self).__init__(self.pretrained_config)
        if self.graph_config.is_train:
            self.pretrain_model = pretrained_model.from_pretrained(graph_config.pretrained_model_name_or_path, config=self.pretrained_config)
            self.pretrain_model.resize_token_embeddings(len(tokenizer))
        else:
            self.pretrain_model = pretrained_model(self.pretrained_config)  # 推理时候只需要加载超参数, 不需要预训练模型的权重
            self.pretrain_model.resize_token_embeddings(len(tokenizer))
        # # tokenizer.model_max_length = self.model.config.max_position_embeddings
        # 如果用隐藏层输出
        if self.graph_config.output_hidden_states:
            # self.dense = FCLayer(int(self.pretrained_config.hidden_size*len(self.graph_config.output_hidden_states)*3), self.graph_config.num_labels,
            #                      is_dropout=self.graph_config.is_dropout, is_active=self.graph_config.is_active, active_type=self.graph_config.active_type)
            self.dense = FCLayer(
                int(self.pretrained_config.hidden_size * len(self.graph_config.output_hidden_states)),
                self.graph_config.num_labels,
                is_dropout=self.graph_config.is_dropout, is_active=self.graph_config.is_active,
                active_type=self.graph_config.active_type)
        else:
            self.dense = FCLayer(self.pretrained_config.hidden_size, self.graph_config.num_labels, is_dropout=self.graph_config.is_dropout,
                                 is_active=self.graph_config.is_active, active_type=self.graph_config.active_type)

        # 池化层
        self.global_maxpooling = torch.nn.AdaptiveMaxPool1d(1)
        self.global_avgpooling = torch.nn.AdaptiveAvgPool1d(1)
        # 激活层/随即失活层
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        output = self.pretrain_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # output, 输出可以自己改
        if self.graph_config.output_hidden_states:
            x = output[2]
            hidden_states_idx = [i for i in range(len(x))]

            ### pool, [max-pool, avg-pool, cls]
            # x_cat = torch.cat([x[i] for i in self.graph_config.output_hidden_states if i in hidden_states_idx], dim=-1)
            # x_max = self.global_maxpooling(x_cat.permute(0, 2, 1)).squeeze(dim=-1)
            # x_avg = self.global_avgpooling(x_cat.permute(0, 2, 1)).squeeze(dim=-1)
            # x_cls = x_cat[:, 0, :]
            # x_merge = torch.cat([x_max, x_avg, x_cls], dim=-1)

            ### cls-concat
            cls = torch.cat([x[i][:, 0, :] for i in self.graph_config.output_hidden_states if i in hidden_states_idx], dim=-1)

            # cls = self.dropout(p=self.graph_config.dropout_rate)(cls)
        else:
            cls = output[0][:, 0, :]  # cls
        logits = self.dense(cls)  # fc
        # inference
        if self.graph_config.is_fc_sigmoid:
            return self.sigmoid(logits)
        elif self.graph_config.is_fc_softmax:
            return self.softmax(logits)
        return logits

