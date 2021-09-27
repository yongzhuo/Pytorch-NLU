# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:35
# @author  : Mo
# @function: graph of pre-train model


from slLayer import PriorMultiLabelSoftMarginLoss, MultiLabelCircleLoss, LabelSmoothingCrossEntropy
from slLayer import CRF, SpanFCLayer, FCLayer, FocalLoss, DiceLoss, Swish, Mish
from slConfig import _SL_MODEL_SOFTMAX, _SL_MODEL_GRID, _SL_MODEL_SPAN, _SL_MODEL_CRF
from slConfig import PRETRAINED_MODEL_CLASSES, _SL_DATA_CONLL, _SL_DATA_SPAN
from slLayer import GridPointer

# torch
from transformers import BertPreTrainedModel
import torch


class Graph(BertPreTrainedModel):
    def __init__(self, graph_config):
        """
        Pytorch Graph of SequenceLabeling, Pre-Trained Model based
        Config:
            config: json, params of graph, eg. {"num_labels":17, "model_type":"BERT"}
        Returns:
            output: Tuple, Tensor of logits and loss
        Url: https://github.com/yongzhuo
        """
        # 预训练语言模型、读取
        self.graph_config = graph_config
        pretrained_config, pretrained_tokenizer, pretrained_model = PRETRAINED_MODEL_CLASSES[graph_config.model_type]
        self.pretrained_config = pretrained_config.from_pretrained(graph_config.pretrained_model_name_or_path, output_hidden_states=graph_config.output_hidden_states)
        self.tokenizer = pretrained_tokenizer.from_pretrained(graph_config.pretrained_model_name_or_path)
        super(Graph, self).__init__(self.pretrained_config)
        self.pretrain_model = pretrained_model.from_pretrained(graph_config.pretrained_model_name_or_path, config=self.pretrained_config)
        self.tokenizer.model_max_length = self.pretrain_model.config.max_position_embeddings
        # 是否软化输出的logits, 否则用label
        if self.graph_config.is_soft_label:
            dim_soft_label = self.graph_config.num_labels
        else:
            dim_soft_label = 1
        # 如果用隐藏层输出, 输出层的选择
        if self.graph_config.output_hidden_states:
            self.fc_span_start = SpanFCLayer(int(self.pretrained_config.hidden_size*len(self.graph_config.output_hidden_states)), self.graph_config.num_labels,
                                    is_active=self.graph_config.is_active, is_dropout=self.graph_config.is_dropout, active_type=self.graph_config.active_type)
            self.fc_span_end = SpanFCLayer(int(self.pretrained_config.hidden_size*len(self.graph_config.output_hidden_states) + dim_soft_label), self.graph_config.num_labels,
                                    is_active=self.graph_config.is_active, is_dropout=self.graph_config.is_dropout, active_type=self.graph_config.active_type)
            self.fc = FCLayer(int(self.pretrained_config.hidden_size*len(self.graph_config.output_hidden_states) + dim_soft_label), self.graph_config.num_labels,
                                    is_active=self.graph_config.is_active, is_dropout=self.graph_config.is_dropout, active_type=self.graph_config.active_type)
            if self.graph_config.task_type.upper() in [_SL_MODEL_GRID]:
                self.fc = torch.nn.Linear(int(self.pretrained_config.hidden_size*len(self.graph_config.output_hidden_states)), self.graph_config.num_labels * self.graph_config.head_size * 2)
        else:
            self.fc_span_start = SpanFCLayer(self.pretrained_config.hidden_size, self.graph_config.num_labels,
                                    is_active=self.graph_config.is_active, is_dropout=self.graph_config.is_dropout, active_type=self.graph_config.active_type)
            self.fc_span_end = SpanFCLayer(self.pretrained_config.hidden_size + dim_soft_label, self.graph_config.num_labels,
                                    is_active=self.graph_config.is_active, is_dropout=self.graph_config.is_dropout, active_type=self.graph_config.active_type)
            self.fc = FCLayer(self.pretrained_config.hidden_size, self.graph_config.num_labels,
                                    is_active=self.graph_config.is_active, is_dropout=self.graph_config.is_dropout, active_type=self.graph_config.active_type)
            if self.graph_config.task_type.upper() in [_SL_MODEL_GRID]:
                self.fc = torch.nn.Linear(self.pretrained_config.hidden_size, self.graph_config.num_labels * self.graph_config.head_size * 2)
        # 网格(全局)指针网络, GPN
        self.layer_grid_pointer = GridPointer(head_nums=self.graph_config.num_labels, head_size=self.graph_config.head_size, is_RoPE=True)
        # 条件随机场层, CRF
        self.layer_crf = CRF(num_tags=self.graph_config.num_labels, batch_first=True)
        # 池化层
        self.global_maxpooling = torch.nn.AdaptiveMaxPool1d(1)
        self.global_avgpooling = torch.nn.AdaptiveAvgPool1d(1)
        # 损失函数, loss
        self.loss_type = self.graph_config.loss_type if self.graph_config.loss_type else "BCE"
        self.loss_mlsm = torch.nn.MultiLabelSoftMarginLoss()  # like BCEWithLogitsLoss
        self.loss_bcelog = torch.nn.BCEWithLogitsLoss()
        self.loss_bce = torch.nn.BCELoss()
        self.loss_mse = torch.nn.MSELoss()
        self.loss_pmlsm = PriorMultiLabelSoftMarginLoss(prior=self.graph_config.prior,num_labels=self.graph_config.num_labels)
        self.loss_lsce = LabelSmoothingCrossEntropy()
        self.loss_circle = MultiLabelCircleLoss()
        self.loss_focal = FocalLoss()
        self.loss_dice = DiceLoss()
        # 激活层/随即失活层
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, labels_start=None, labels_end=None):
        output = self.pretrain_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # BERT输出多层拼接
        if self.graph_config.output_hidden_states:
            x = output[2]
            hidden_states_idx = [i for i in range(len(x))]
            bert_sequence = torch.cat([x[i] for i in self.graph_config.output_hidden_states if i in hidden_states_idx], dim=-1)
        else:
            bert_sequence = output[0]  # shape <32, 128, 768>
        # bert后接的层等
        if self.graph_config.task_type.upper() in [_SL_MODEL_SPAN]:
            logits_start_org = self.fc_span_start(bert_sequence)
            if labels_start is not None and self.graph_config.is_train:
                if self.graph_config.is_soft_label:
                    label_logits = torch.Tensor(input_ids.size(0), input_ids.size(1), self.graph_config.num_labels).to(input_ids.device).zero_().float()
                    label_logits.scatter_(2, labels_start.unsqueeze(2).long(), 1).float()
                else:
                    label_logits = labels_start.unsqueeze(2).float()
            else:
                label_logits = self.softmax(logits_start_org)
                if not self.graph_config.is_soft_label:
                    label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
            cls_extend_start = torch.cat([bert_sequence, label_logits], dim=-1)
            logits_end_org = self.fc_span_end(cls_extend_start)
        elif self.graph_config.task_type.upper() in [_SL_MODEL_GRID]:  # Grid-Pointer-Network
            logits_fc = self.fc(bert_sequence)
            logits_org = self.layer_grid_pointer(logits_fc, attention_mask, token_type_ids)
        else:
            logits_org = self.fc(bert_sequence)  # full-connect: FCLayer
        loss = None
        # 训练阶段的输出
        if labels is not None or labels_start is not None:  # loss
            # 只取attention_mask的内容
            if attention_mask is not None:
                att_mask = attention_mask.view(-1) == 1
            if   self.graph_config.task_type.upper() in [_SL_MODEL_SOFTMAX]:
                logits = logits_org.view(-1, self.graph_config.num_labels)[att_mask]
                labels = labels.view(-1)[att_mask]
                labels = torch.zeros(logits.shape[0], self.graph_config.num_labels) \
                    .to(labels.device).scatter_(1, labels.unsqueeze(1).long(), 1)
            elif self.graph_config.task_type.upper() in [_SL_MODEL_SPAN]:
                # 片段SPAN损失
                logits_start = logits_start_org.view(-1, self.graph_config.num_labels)[att_mask]
                logits_end = logits_end_org.view(-1, self.graph_config.num_labels)[att_mask]
                labels_start = labels_start.view(-1)[att_mask]
                labels_end = labels_end.view(-1)[att_mask]
                # labels转化为onehot的形式, 方便使用BCE等方法
                labels_start = torch.zeros(labels_start.unsqueeze(1).shape[0], self.graph_config.num_labels) \
                    .to(input_ids.device).scatter_(1, labels_start.unsqueeze(1).long(), 1)
                labels_end = torch.zeros(labels_end.unsqueeze(1).shape[0], self.graph_config.num_labels) \
                    .to(input_ids.device).scatter_(1, labels_end.unsqueeze(1).long(), 1)
                # 拼接方法导致loss-start/loss-end权重变成不可选择, 都是0.5
                logits = torch.cat([logits_start, logits_end], 1)  # <32, 128+128, 7>
                labels = torch.cat([labels_start, labels_end], 1)
            elif self.graph_config.task_type.upper() in [_SL_MODEL_GRID]:
                # 网格(全局)Grid损失
                logits = logits_org.view(-1)
                labels = labels.view(-1)
            # Loss
            if self.graph_config.task_type.upper() in [_SL_MODEL_CRF]:  # 条件随机场CRF损失
                loss = self.layer_crf(emissions=logits_org, tags=labels.long(), mask=attention_mask)
                logits = logits_org
                loss = - loss
            elif self.loss_type.upper() == "PRIOR_MARGIN_LOSS":# 带先验的边缘损失
                loss = self.loss_pmlsm(logits, labels)
            elif self.loss_type.upper() == "SOFT_MARGIN_LOSS": # 边缘损失pytorch版, 划分距离
                loss = self.loss_mlsm(logits, labels)
            elif self.loss_type.upper() == "FOCAL_LOSS":       # 聚焦损失(学习难样本, 2-0.25, 负样本多的情况)
                loss = self.loss_focal(logits, labels)
            elif self.loss_type.upper() == "CIRCLE_LOSS":      # 圆形损失(均衡, 统一 triplet-loss 和 softmax-ce-loss)
                loss = self.loss_circle(logits, labels)
            elif self.loss_type.upper() == "DICE_LOSS":        # 切块损失(图像)
                loss = self.loss_dice(logits, labels.long())
            elif self.loss_type.upper() == "LABEL_SMOOTH":     # 交叉熵平滑
                loss = self.loss_lsce(logits, labels.long())
            elif self.loss_type.upper() == "BCE_LOGITS":       # 二元交叉熵平滑连续计算型pytorch版
                loss = self.loss_bcelog(logits, labels)
            elif self.loss_type.upper() == "MSE":              # 均方误差
                loss = self.loss_mse(logits.view(-1), labels.view(-1))
            elif self.loss_type.upper() == "BCE":              # 二元交叉熵的pytorch版
                logits_softmax = self.softmax(logits)
                loss = self.loss_bce(logits_softmax.view(-1), labels.view(-1))
            elif self.loss_type.upper() == "MIX":              # 混合误差[聚焦损失/2 + 带先验的边缘损失/2]
                loss_focal = self.loss_focal(logits.view(-1), labels.view(-1))
                loss_pmlsm = self.loss_pmlsm(logits, labels)
                loss = (loss_pmlsm + loss_focal) / 2
            else:                                              # 其他情况: 二元交叉熵
                logits_softmax = self.softmax(logits)
                loss = self.loss_bce(logits_softmax.view(-1), labels.view(-1))
        # 预测阶段的输出等
        if self.graph_config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_GRID] and not self.graph_config.is_train:
            logits = logits_org  # SL-SOFTMAX
        if self.graph_config.task_type.upper() in [_SL_MODEL_SPAN] and not self.graph_config.is_train:
            logits = torch.cat([logits_start_org, logits_end_org], 1)  # <32, 128+128, 7>
        if self.graph_config.task_type.upper() in [_SL_MODEL_CRF] and not self.graph_config.is_train:
            logits = self.layer_crf.decode(logits_org, attention_mask).squeeze(0)  # CRF-Decode, 即标签tags
        return loss, logits

