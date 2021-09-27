# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:35
# @author  : Mo
# @function: graph of pre-train model


from tcLayer import PriorMultiLabelSoftMarginLoss, MultiLabelCircleLoss, LabelSmoothingCrossEntropy
from tcLayer import FCLayer, FocalLoss, DiceLoss
from tcConfig import PRETRAINED_MODEL_CLASSES
# torch
from transformers import BertPreTrainedModel
import torch


class TCGraph(BertPreTrainedModel):
    def __init__(self, graph_config):
        """
        Pytorch Graph of TextClassification, Pre-Trained Model based
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
        self.tokenizer = pretrained_tokenizer.from_pretrained(graph_config.pretrained_model_name_or_path)
        super(TCGraph, self).__init__(self.pretrained_config)
        self.model = pretrained_model.from_pretrained(graph_config.pretrained_model_name_or_path, config=self.pretrained_config)
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings
        # 如果用隐藏层输出
        if self.graph_config.output_hidden_states:
            self.dense = FCLayer(int(self.pretrained_config.hidden_size*len(self.graph_config.output_hidden_states)*3), self.graph_config.num_labels,
                                 is_dropout=self.graph_config.is_dropout, is_active=self.graph_config.is_active, active_type=self.graph_config.active_type)
        else:
            self.dense = FCLayer(self.pretrained_config.hidden_size, self.graph_config.num_labels, is_dropout=self.graph_config.is_dropout,
                                 is_active=self.graph_config.is_active, active_type=self.graph_config.active_type)

        # 池化层
        self.global_maxpooling = torch.nn.AdaptiveMaxPool1d(1)
        self.global_avgpooling = torch.nn.AdaptiveAvgPool1d(1)
        # 损失函数, loss
        self.loss_type = self.graph_config.loss_type if self.graph_config.loss_type else "BCE"
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.loss_mlsm = torch.nn.MultiLabelSoftMarginLoss()  # like BCEWithLogitsLoss
        self.loss_bcelog = torch.nn.BCEWithLogitsLoss()
        self.loss_bce = torch.nn.BCELoss()
        self.loss_mse = torch.nn.MSELoss()
        self.loss_pmlsm = PriorMultiLabelSoftMarginLoss(prior=self.graph_config.prior, num_labels=self.graph_config.num_labels)
        self.loss_circle = MultiLabelCircleLoss()
        self.loss_lsce = LabelSmoothingCrossEntropy()
        self.loss_focal = FocalLoss()
        self.loss_dice = DiceLoss()
        # 激活层/随即失活层
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.graph_config.output_hidden_states:
            x = output[2]
            hidden_states_idx = [i for i in range(len(x))]
            x_cat = torch.cat([x[i] for i in self.graph_config.output_hidden_states if i in hidden_states_idx], dim=-1)
            #  pool, [max-pool, avg-pool, cls]
            x_max = self.global_maxpooling(x_cat.permute(0, 2, 1)).squeeze(dim=-1)
            x_avg = self.global_avgpooling(x_cat.permute(0, 2, 1)).squeeze(dim=-1)
            x_cls = x_cat[:, 0, :]
            x_merge = torch.cat([x_max, x_avg, x_cls], dim=-1)
            cls = self.dropout(p=self.graph_config.dropout_rate)(x_merge)
        else:
            cls = output[0][:, 0, :]  # cls
        logits = self.dense(cls)  # full-connect: FCLayer
        loss = None
        if labels is not None:  # loss
            if self.loss_type.upper() == "PRIOR_MARGIN_LOSS":  # 带先验的边缘损失
                loss = self.loss_pmlsm(logits, labels)
            elif self.loss_type.upper() == "SOFT_MARGIN_LOSS": # 边缘损失pytorch版, 划分距离
                loss = self.loss_mlsm(logits, labels)
            elif self.loss_type.upper() == "FOCAL_LOSS":       # 聚焦损失(学习难样本, 2-0.25, 负样本多的情况)
                loss = self.loss_focal(logits.view(-1), labels.view(-1))
            elif self.loss_type.upper() == "CIRCLE_LOSS":      # 圆形损失(均衡, 统一 triplet-loss 和 softmax-ce-loss)
                loss = self.loss_circle(logits, labels)
            elif self.loss_type.upper() == "DICE_LOSS":        # 切块损失(图像)
                loss = self.loss_dice(logits, labels.long())
            elif self.loss_type.upper() == "LABEL_SMOOTH":     # 交叉熵平滑
                loss = self.loss_lsce(logits, labels.long())
            elif self.loss_type.upper() == "BCE_LOGITS":       # 二元交叉熵平滑连续计算型pytorch版
                loss = self.loss_bcelog(logits, labels)
            elif self.loss_type.upper() == "BCE":              # 二元交叉熵的pytorch版
                logits_softmax = self.softmax(logits)
                loss = self.loss_bce(logits_softmax.view(-1), labels.view(-1))
            elif self.loss_type.upper() == "MSE":              # 均方误差
                loss = self.loss_mse(logits.view(-1), labels.view(-1))
            elif self.loss_type.upper() == "MIX":              # 混合误差[聚焦损失/2 + 带先验的边缘损失/2]
                loss_focal = self.loss_focal(logits.view(-1), labels.view(-1))
                loss_pmlsm = self.loss_pmlsm(logits, labels)
                loss = (loss_pmlsm + loss_focal) / 2
            else:                                              # 二元交叉熵
                logits_softmax = self.softmax(logits)
                loss = self.loss_bce(logits_softmax.view(-1), labels.view(-1))
        return loss, logits

