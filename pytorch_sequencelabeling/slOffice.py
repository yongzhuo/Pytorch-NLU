# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:36
# @author  : Mo
# @function: office of transformers


from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
import torch

from slTools import chinese_extract_extend, get_pos_from_common, get_pos_from_span, mertics_report_sequence_labeling
from slConfig import _SL_MODEL_SOFTMAX, _SL_MODEL_GRID, _SL_MODEL_SPAN, _SL_MODEL_CRF
from slConfig import _SL_DATA_CONLL, _SL_DATA_SPAN
from slAdversarial import FGM
from slGraph import Graph

from slTqdm import tqdm, trange
import logging as logger
import numpy as np
import random
import codecs
import json
import os


class Office:
    def __init__(self, config, train_corpus=None, dev_corpus=None, tet_corpus=None, logger=logger):
        """
        初始化主训练器/算法网络架构/数据集, init Trainer
        config:
            config: json, params of graph, eg. {"num_labels":17, "model_type":"BERT"}
            train_corpus: List, train corpus of dataset
            dev_corpus: List, dev corpus of dataset 
            tet_corpus: List, tet corpus of dataset 
        Returns:
            None
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and self.config.is_cuda else "cpu"
        self.model = Graph(config).to(self.device)  # 初始化模型网络架构
        self.set_random_seed(self.config.seed)  # 初始化随机种子
        self.train_corpus = train_corpus
        self.dev_corpus = dev_corpus
        self.tet_corpus = tet_corpus
        self.logger = logger
        self.logger.info(config)
        self.rounded = 4

    def set_random_seed(self, seed):
        """
        初始化随机种子, init seed
        config:
            seed: int, seed of all, eg. 2021
        Returns:
            None
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.config.is_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def evaluate(self, mode_type):
        """
        验证, evalate
        config:
            mode_type: str, eg. "tet", "dev"
        Returns:
            result: json
        """
        if   mode_type == "tet":
            corpus, texts = self.tet_corpus
        elif mode_type == "dev":
            corpus, texts = self.dev_corpus
        else:
            raise Exception("mode_type must be 'dev' or 'tet'")

        data_loader = DataLoader(corpus, batch_size=self.config.batch_size)
        ys_pred_id, ys_true_id = None, None
        eval_loss = 0.0
        eval_steps = 0
        # 验证
        for batch_data in tqdm(data_loader, desc="evaluate"):
            batch_data = [bd.to(self.device) for bd in batch_data]  # device
            headers = ["input_ids", "attention_mask", "token_type_ids", "labels", "labels_start", "labels_end"]  # 必须按照顺序
            if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF, _SL_MODEL_GRID]:  # CRF or softmax
                inputs = dict(zip(headers[:-2], batch_data))
            elif self.config.task_type.upper() in [_SL_MODEL_SPAN]:  # SPAN
                inputs = dict(zip(headers[:-3] + headers[-2:], batch_data))
            else:
                raise ValueError("invalid data_loader, length of batch_data must be 4 or 6!")
            with torch.no_grad():
                output = self.model(**inputs)
                loss, logits = output[:2]
                eval_loss += loss.item()
                eval_steps += 1
            # 获取numpy格式label/logits
            if self.config.task_type.upper() in [_SL_MODEL_SPAN]:
                labels_start = inputs.get("labels_start").detach().cpu().numpy()
                labels_end = inputs.get("labels_end").detach().cpu().numpy()
                logits_numpy = logits.detach().cpu().numpy()
                inputs_numpy = np.concatenate([labels_start, labels_end], axis=1)
            else:
                inputs_numpy = inputs.get("labels").detach().cpu().numpy()
                logits_numpy = logits.detach().cpu().numpy()
            # 追加, extend
            if ys_pred_id is not None:
                ys_pred_id = np.append(ys_pred_id, logits_numpy, axis=0)
                ys_true_id = np.append(ys_true_id, inputs_numpy, axis=0)
            else:
                ys_pred_id = logits_numpy
                ys_true_id = inputs_numpy
        eval_loss = eval_loss / eval_steps
        # 最后输出, top-1
        if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF]:
            ys_pred_id = np.argmax(ys_pred_id, axis=-1) if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX] else ys_pred_id
            ys_true_id = ys_true_id.tolist()
            ys_pred_id = ys_pred_id.tolist()
            res_myz_true = []
            res_myz_pred = []
            count = 0
            for x, y, z in zip(ys_pred_id, ys_true_id, texts[:len(ys_pred_id)]):
                len_text = len(z)
                x1 = [self.config.i2l[str(int(xi))] for xi in x[1:-1]][:len_text]
                y1 = [self.config.i2l[str(int(yi))] for yi in y[1:-1]][:len_text]
                pos_pred = get_pos_from_common(z, x1)
                pos_true = get_pos_from_common(z, y1)
                res_myz_pred += [{"text":z, "label":pos_pred}]
                res_myz_true += [{"text":z, "label":pos_true}]
                if count < 5:
                    self.logger.info("y_pred:{}".format(x1))
                    self.logger.info("y_true:{}".format(y1))
                    self.logger.info("pos_pred:{}".format(pos_pred))
                    self.logger.info("pos_true:{}".format(pos_true))
                count += 1
            i2l = {}
            label_set = set()
            for i,lab in self.config.i2l.items():
                lab_1 = lab.split("-")[-1]
                if lab_1 not in label_set:
                    i2l[str(len(i2l))] = lab_1
                    label_set.add(lab_1)
        elif self.config.task_type.upper() in [_SL_MODEL_SPAN]:
            res_myz_true = []
            res_myz_pred = []
            count = 0
            for logits, labels, text_i in zip(ys_pred_id, ys_true_id, texts[:len(ys_pred_id)]):
                len_text = len(text_i)
                len_logits = logits.shape[0]
                logits_start = logits[:int(len_logits / 2)][1:-1][:len_text]
                logits_end = logits[int(len_logits / 2):][1:-1][:len_text]
                labels_start = labels[:int(len_logits / 2)][1:-1][:len_text]
                labels_end = labels[int(len_logits / 2):][1:-1][:len_text]
                pos_logits = get_pos_from_span(logits_start.tolist(), logits_end.tolist(), self.config.i2l)
                pos_true, pos_pred = [], []
                for ps_i in pos_logits:
                    pos_pred.append({"type": ps_i[0], "pos": [ps_i[1], ps_i[2]], "ent": text_i[ps_i[1]:ps_i[2]+1]})
                pos_label = get_pos_from_span(labels_start.tolist(), labels_end.tolist(), self.config.i2l, use_index=True)
                for ps_i in pos_label:
                    pos_true.append({"type": ps_i[0], "pos": [ps_i[1], ps_i[2]], "ent": text_i[ps_i[1]:ps_i[2]+1]})
                res_myz_pred += [{"text": text_i, "label": pos_pred}]
                res_myz_true += [{"text": text_i, "label": pos_true}]
                if count < 5:
                    self.logger.info("pos_pred:{}".format(pos_pred))
                    self.logger.info("pos_true:{}".format(pos_true))
                count += 1
            i2l = self.config.i2l
        elif self.config.task_type.upper() in [_SL_MODEL_GRID]:
            ys_true_id = ys_true_id.tolist()
            ys_pred_id = ys_pred_id.tolist()
            res_myz_true = []
            res_myz_pred = []
            count = 0
            for x, y, z in zip(ys_pred_id, ys_true_id, texts[:len(ys_pred_id)]):
                pos_pred, pos_true = [], []
                x,y = np.array(x), np.array(y)
                x[:, [0, -1]] -= np.inf
                x[:, :, [0, -1]] -= np.inf
                y[:, [0, -1]] -= np.inf
                y[:, :, [0, -1]] -= np.inf
                for pos_type, pos_start, pos_end in zip(*np.where(x > self.config.grid_pointer_threshold)):
                    pos_start, pos_end = pos_start-1, pos_end-1
                    pos_type = self.config.i2l[str(int(pos_type))]
                    if pos_type != "O":
                        line = {"type": pos_type, "pos": [pos_start, pos_end], "ent": z[pos_start:pos_end+1]}
                        pos_pred.append(line)
                for pos_type, pos_start, pos_end in zip(*np.where(y > self.config.grid_pointer_threshold)):
                    pos_start, pos_end = pos_start-1, pos_end-1
                    pos_type = self.config.i2l[str(int(pos_type))]
                    if pos_type != "O":
                        line = {"type": pos_type, "pos": [pos_start, pos_end], "ent": z[pos_start:pos_end+1]}
                        pos_true.append(line)
                res_myz_pred += [{"text":z, "label":pos_pred}]
                res_myz_true += [{"text":z, "label":pos_true}]
                if count < 5:
                    # self.logger.info("y_pred:{}".format(x))
                    # self.logger.info("y_true:{}".format(y))
                    self.logger.info("pos_pred:{}".format(pos_pred[:5]))
                    self.logger.info("pos_true:{}".format(pos_true[:5]))
                count += 1
            i2l = self.config.i2l

        # 评估
        self.logger.info("i2l:{}".format(i2l))
        mertics_dict, mertics_report, mcm_report, y_error_dict = mertics_report_sequence_labeling(res_myz_true, res_myz_pred, i2l)
        self.logger.info(y_error_dict[:5])
        # self.logger.info(res_myz_pred[:5])
        self.logger.info("confusion_matrix:\n " + mcm_report)
        self.logger.info("mertics: \n" + mertics_report)
        result = {"loss": eval_loss}
        result.update(mertics_dict)
        return result, mertics_report

    def predict(self, corpus):
        """
        预测, pred
        config:
            corpus: tensor, eg. tensor(1,2,3)
        Returns:
            ys_prob: list<json>
        """
        corpus_p, texts = corpus
        data_loader = DataLoader(corpus_p, batch_size=self.config.batch_size, pin_memory=True)
        ys_pred_id = None
        #  预测 batch-size
        self.model.eval()
        for batch_data in data_loader:
            batch_data = [bd.to(self.device) for bd in batch_data]  # device
            headers = ["input_ids", "attention_mask", "token_type_ids", "labels", "labels_start", "labels_end"]  # 必须按照顺序
            inputs = dict(zip(headers[:-3], batch_data))
            with torch.no_grad():
                output = self.model(**inputs)
                loss, logits = output[:2]
            # 获取numpy格式label/logits
            if self.config.task_type.upper() in [_SL_MODEL_SPAN]:
                logits_numpy = logits.detach().cpu().numpy()
            else:
                logits_numpy = logits.detach().cpu().numpy()
            # 追加, extend
            if ys_pred_id is not None:
                ys_pred_id = np.append(ys_pred_id, logits_numpy, axis=0)
            else:
                ys_pred_id = logits_numpy
        res_myz_pred = []
        # 最后输出, top-1
        if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF]:
            ys_pred_id = np.argmax(ys_pred_id, axis=-1) if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX] else ys_pred_id
            ys_pred_id = ys_pred_id.tolist()
            # count = 0
            for x, z in zip(ys_pred_id, texts[:len(ys_pred_id)]):
                len_text = len(z)
                x1 = [self.config.i2l[str(int(xi))] for xi in x[1:-1]][:len_text]
                pos_pred = get_pos_from_common(z, x1)
                res_myz_pred += [{"label": pos_pred, "text": z}]
                # if count < 5:
                #     self.logger.info("y_pred:{}".format(x1))
                #     self.logger.info("pos_pred:{}".format(pos_pred))
                # count += 1
        elif self.config.task_type.upper() in [_SL_MODEL_SPAN]:
            # count = 0
            for logits, text_i in zip(ys_pred_id, texts[:len(ys_pred_id)]):
                len_text = len(text_i)
                len_logits = logits.shape[0]
                logits_start = logits[:int(len_logits / 2)][1:-1][:len_text]
                logits_end = logits[int(len_logits / 2):][1:-1][:len_text]
                pos_logits = get_pos_from_span(logits_start.tolist(), logits_end.tolist(), self.config.i2l)
                pos_true, pos_pred = [], []
                for ps_i in pos_logits:
                    pos_pred.append({"type": ps_i[0], "pos": [ps_i[1], ps_i[2]], "ent": text_i[ps_i[1]:ps_i[2] + 1]})
                res_myz_pred += [{"label": pos_pred, "text": text_i}]
                # if count < 5:
                #     self.logger.info("pos_pred:{}".format(pos_pred))
                # count += 1
        elif self.config.task_type.upper() in [_SL_MODEL_GRID]:
            # count = 0
            for logits, text_i in zip(ys_pred_id, texts[:len(ys_pred_id)]):
                pos_pred = []
                logits = np.array(logits)
                logits[:, [0, -1]] -= np.inf
                logits[:, :, [0, -1]] -= np.inf
                for pos_type, pos_start, pos_end in zip(*np.where(logits > self.config.grid_pointer_threshold)):
                    pos_start, pos_end = pos_start-1, pos_end-1
                    pos_type = self.config.i2l.get(str(int(pos_type)), "")
                    if pos_type != "O":
                        line = {"type": pos_type, "pos": [pos_start, pos_end], "ent": text_i[pos_start:pos_end + 1]}
                        pos_pred.append(line)
                res_myz_pred += [{"label": pos_pred, "text": text_i}]
                # if count < 5:
                #     self.logger.info("pos_pred:{}".format(pos_pred))
                # count += 1
        return res_myz_pred

    def train_model(self):
        """  训练迭代epoch  
             return global_steps, best_mertics
        """
        #  数据转为迭代器iter的形式    #  weight_decay_rate=0.01, grad_accum_steps
        train_data, _ = self.train_corpus
        data_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=self.config.batch_size)

        #  配置好优化器与训练工作计划(主要是学习率预热warm-up与衰减weight-decay, 以及不作用的层超参)
        no_decay = ["LayerNorm.weight", "bias"]
        pretrain_params = list(self.model.pretrain_model.named_parameters())
        if      self.config.task_type.upper() in [_SL_MODEL_SPAN]:
            fc_params = list(self.model.fc_span_start.named_parameters()) + list(self.model.fc_span_end.named_parameters())
        elif self.config.task_type.upper() in [_SL_MODEL_SOFTMAX]:
            fc_params = list(self.model.fc.named_parameters())
        else:
            fc_params = list(self.model.layer_crf.named_parameters())
        parameters_no_decay = [
            {"params": [p for n, p in pretrain_params if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config.weight_decay, "lr": self.config.lr},
            {"params": [p for n, p in pretrain_params if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, "lr": self.config.lr},

            {"params": [p for n, p in fc_params if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config.weight_decay, "lr": self.config.dense_lr},
            {"params": [p for n, p in fc_params if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, "lr": self.config.dense_lr}
        ]

        optimizer = AdamW(parameters_no_decay, lr=self.config.lr, eps=self.config.adam_eps)
        # 训练轮次
        times_batch_size = len(data_loader) // self.config.grad_accum_steps
        num_training_steps = int(times_batch_size * self.config.epochs)
        # 如果选择-1不设置则为 半个epoch
        num_warmup_steps = int((len(data_loader) // self.config.grad_accum_steps // 2)) if self.config.warmup_steps == -1 else self.config.warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        tensorboardx_witer = SummaryWriter(logdir=self.config.model_save_path)

        # adv
        if self.config.is_adv:
            fgm = FGM(self.model, emb_name=self.config.adv_emb_name, epsilon=self.config.adv_eps)
        # 开始训练
        epochs_store = []
        global_steps = 0
        best_mertics = {}
        best_report = ""
        for epochs_i in trange(self.config.epochs, desc="epoch"):  # epoch
            for idx, batch_data in enumerate(tqdm(data_loader, desc="step")):  # step
                # 数据与模型, 获取输入的json
                batch_data = [bd.to(self.device) for bd in batch_data]  # device
                # warning: 必须按照该顺序zip
                # SPAN:  tensor_input, tensor_attention_mask, tensor_token_type, tensor_start, tensor_end
                # CRF-SOFTMAX-GRID:  tensor_input, tensor_attention_mask, tensor_token_type, tensor_label
                headers = ["input_ids", "attention_mask", "token_type_ids", "labels", "labels_start", "labels_end"]
                if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF, _SL_MODEL_GRID]:  # CRF or softmax
                    inputs = dict(zip(headers[:-2], batch_data))
                elif self.config.task_type.upper() in [_SL_MODEL_SPAN]:  # SPAN
                    inputs = dict(zip(headers[:-3] + headers[-2:], batch_data))
                else:
                    raise ValueError("invalid data_loader, length of batch_data must be 4 or 6!")
                # model
                outputs = self.model(**inputs)
                loss = outputs[0] / self.config.grad_accum_steps
                loss.backward()
                global_steps += 1
                #  对抗训练
                if self.config.is_adv:
                    fgm.attack()  # 在embedding上添加对抗扰动
                    outputs = self.model(**inputs)
                    loss = outputs[0] / self.config.grad_accum_steps  # 梯度累计
                    loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数
                #  梯度累计
                if (idx + 1) % self.config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                # 评估算法/打印日志/存储模型, 1个epoch/到达保存的步数/或者是最后一轮最后一步
                if (self.config.evaluate_steps > 0 and global_steps % self.config.evaluate_steps == 0) or (idx == times_batch_size-1) \
                        or (epochs_i+1 == self.config.epochs and idx+1 == len(data_loader)):
                    # 评估, is_train要置为False
                    self.model.graph_config.is_train = False
                    self.model.eval()
                    res, report = self.evaluate("dev")
                    self.logger.info("epoch_global: {}, step_global: {}, step: {}".format(epochs_i, global_steps, idx))
                    self.logger.info("best_report:\n" + best_report)
                    # self.logger.info("current_mertics:\n {}".format(res))
                    # idx_score = res.get("micro", {}).get("f1", 0)  # "macro", "micro", "weighted"
                    for k,v in res.items():  # tensorboard日志, 其中抽取中文、数字和英文, 避免一些不支持的符号, 比如说 /\|等特殊字符
                        if type(v) == dict:  # 空格和一些特殊字符tensorboardx.add_scalar不支持
                            k = chinese_extract_extend(k)
                            k = k.replace(" ", "")
                            for k_2, v_2 in v.items():
                                tensorboardx_witer.add_scalar(k + "/" + k_2, v_2, global_steps)
                        elif type(v) == float:
                            tensorboardx_witer.add_scalar(k, v, global_steps)
                            tensorboardx_witer.add_scalar("lr", scheduler.get_lr()[-1], global_steps)  #  pytorch==1.4.0版本往后: scheduler.get_last_lr()
                    save_best_mertics_key = self.config.save_best_mertics_key  # 模型存储的判别指标
                    abmk_1 = save_best_mertics_key[0]  # like "micro_avg"
                    abmk_2 = save_best_mertics_key[1]  # like "f1-score"
                    if res.get(abmk_1, {}).get(abmk_2, 0) > best_mertics.get(abmk_1, {}).get(abmk_2, 0) or best_mertics.get(abmk_1, {}).get(abmk_2, 0)==0.:  # 只保留最优的指标
                        epochs_store.append((epochs_i, idx))
                        res["total"] = {"epochs": epochs_i, "global_steps": global_steps, "step_current": idx}
                        best_mertics = res
                        best_report = report
                        self.save_model()
                    # 重置is_train为True
                    self.model.graph_config.is_train = True  # is_train
                    self.model.train()  # train-type
                    # 早停, 连续stop_epochs指标不增长则自动停止
                    if epochs_store and epochs_i - epochs_store[-1][0] >= self.config.stop_epochs:
                        break
        return global_steps, best_mertics, best_report

    def load_model(self):
        """  加载模型  """
        try:
            path_model = os.path.join(self.config.model_save_path, self.config.model_name)
            self.model = torch.load(path_model)  # , map_location=torch.device(self.device))
            self.model.to(self.device)
            logger.info("******model loaded success******")
        except:
            raise Exception("******load model error******")

    def save_model(self):
        """  存储模型  """
        if not os.path.exists(self.config.model_save_path):
            os.makedirs(self.config.model_save_path)
        # save config
        self.config.is_train = False  # 存储后的用于预测
        path_config = os.path.join(self.config.model_save_path, self.config.config_name)
        with codecs.open(filename=path_config, mode="w", encoding="utf-8") as fc:
            json.dump(vars(self.config), fc, indent=4, ensure_ascii=False)
            fc.close()
        # save model
        path_model = os.path.join(self.config.model_save_path, self.config.model_name)
        torch.save(self.model, path_model)
        logger.info("******model_save_path is {}******".format(path_model))
        self.config.is_train = True  # 如果之后还需要预测的, 因为self.model.graph_config.is_train=self.config.is_train

