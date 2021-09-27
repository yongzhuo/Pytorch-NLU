# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:36
# @author  : Mo
# @function: office of transformers, 训练-主工作流


from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
import torch

from tcTools import chinese_extract_extend, mertics_report, sigmoid, softmax
from tcConfig import _TC_MULTI_CLASS, _TC_MULTI_LABEL
from tcGraph import TCGraph as Graph
from tcTqdm import tqdm, trange
from tcAdversarial import FGM

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
        if mode_type == "test":
            corpus = self.tet_corpus
        elif mode_type == "dev":
            corpus = self.dev_corpus
        else:
            raise Exception("mode_type must be 'dev' or 'tet'")

        data_loader = DataLoader(corpus, batch_size=self.config.batch_size, pin_memory=True)
        ys_pred_id, ys_true_id = None, None
        eval_loss = 0.0
        eval_steps = 0
        self.model.eval()  # 验证
        for batch_data in tqdm(data_loader, desc="evaluate"):
            batch_data = [bd.to(self.device) for bd in batch_data]  # device
            with torch.no_grad():
                inputs = {"attention_mask": batch_data[1],
                          "token_type_ids": batch_data[2],
                          "input_ids": batch_data[0],
                          "labels": batch_data[3],
                          }
                output = self.model(**inputs)
                loss, logits = output[:2]
                eval_loss += loss.mean().item()
                eval_steps += 1

            inputs_numpy = inputs.get("labels").detach().cpu().numpy()
            logits_numpy = logits.detach().cpu().numpy()
            if ys_pred_id is not None:
                ys_pred_id = np.append(ys_pred_id, logits_numpy, axis=0)
                ys_true_id = np.append(ys_true_id, inputs_numpy, axis=0)
            else:
                ys_pred_id = logits_numpy
                ys_true_id = inputs_numpy
        eval_loss = eval_loss / eval_steps
        # 最后输出, top-1
        ys_true_str, ys_pred_str = [], []
        for i in range(ys_pred_id.shape[0]):
            yti = ys_true_id[i]
            ypi = ys_pred_id[i]
            if _TC_MULTI_LABEL == self.config.task_type.upper():  # 多标签分类, 大于阈值该类别职位1, 否则为0
                ypi[ypi >= self.config.multi_label_threshold] = 1
                ypi[ypi < self.config.multi_label_threshold] = 0
            else:  # 多类分类, 最大得分置为1, 非最大得分全部置为0
                index = ypi.argmax()
                ypi[:] = 0
                ypi[index] = 1
            ys_true_str.append(yti)
            ys_pred_str.append(ypi)
        # 评估
        target_names = [self.config.i2l[str(i)] for i in range(len(self.config.i2l))]
        mertics, report = mertics_report(ys_true_str, ys_pred_str, target_names=target_names)
        self.logger.info(report)
        result = {"loss": eval_loss}
        result.update(mertics)
        return result, report

    def predict(self, corpus, rounded=4, logits_type="logits"):
        """
        预测, pred
        config:
            corpus     : tensor, eg. tensor(1,2,3)
            rounded    : int,   eg. 4
            logits_type: str,   eg. "logits", "sigmoid", "softmax"
        Returns:
            ys_prob: list<json>
        """
        # pin_memory预先加载到cuda-gpu
        data_loader = DataLoader(corpus, batch_size=self.config.batch_size, pin_memory=True)
        ys_pred_id = None
        # 预测 batch-size
        self.model.eval()
        for batch_data in data_loader:
            batch_data = [bd.to(self.device) for bd in batch_data]  # device
            with torch.no_grad():
                inputs = {"attention_mask": batch_data[1],
                          "token_type_ids": batch_data[2],
                          "input_ids": batch_data[0],
                          "labels": batch_data[3],
                          }
                output = self.model(**inputs)
                loss, logits = output[:2]
            logits_numpy = logits.detach().cpu().numpy()

            if logits_type.upper() == "SIGMOID":
                logits_numpy = sigmoid(logits_numpy)
            elif logits_type.upper() == "SOFTMAX":
                logits_numpy = softmax(logits_numpy)

            if ys_pred_id:
                ys_pred_id = np.append(ys_pred_id, logits_numpy, axis=0)
            else:
                ys_pred_id = logits_numpy
        # 最大概率, 最大概率的类别, 类别-概率, like [{"label_1":0.8, "label_2":0.2}]
        ys_prob = []
        for i in range(ys_pred_id.shape[0]):
            ypi = ys_pred_id[i].tolist()
            line = {}
            for idx, prob in enumerate(ypi):
                line[self.config.i2l[str(idx)]] = round(prob, rounded)
            ys_prob.append(line)
        return ys_prob
    
    def train_model(self):
        """  训练迭代epoch  
        return global_steps, best_mertics
        """
        #  数据转为迭代器iter的形式    #  weight_decay_rate=0.01, grad_accum_steps
        data_loader = DataLoader(self.train_corpus, sampler=RandomSampler(self.train_corpus), batch_size=self.config.batch_size, pin_memory=True)

        #  配置好优化器与训练工作计划(主要是学习率预热warm-up与衰减weight-decay, 以及不作用的层超参)
        params_no_decay = ["LayerNorm.weight", "bias"]
        parameters_no_decay = [
            {"params": [p for n, p in self.model.named_parameters() if not any(pnd in n for pnd in params_no_decay)],
             "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(pnd in n for pnd in params_no_decay)],
             "weight_decay": 0.0}
            ]
        optimizer = AdamW(parameters_no_decay, lr=self.config.lr, eps=self.config.adam_eps)
        # 训练轮次
        times_batch_size = len(data_loader) // self.config.grad_accum_steps
        num_training_steps = int(times_batch_size * self.config.epochs)
        # 如果选择-1不设置则为 半个epoch
        num_warmup_steps = int((len(data_loader) // self.config.grad_accum_steps // 2)) if self.config.warmup_steps == -1 else self.config.grad_accum_steps
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
            self.model.train()  # train-type
            for idx, batch_data in enumerate(tqdm(data_loader, desc="step")):  # step
                # 数据与模型
                batch_data = [bd.to(self.device) for bd in batch_data]  # device
                inputs = {"attention_mask": batch_data[1],
                          "token_type_ids": batch_data[2],
                          "input_ids": batch_data[0],
                          "labels": batch_data[3],
                          }
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
                    res, report = self.evaluate("dev")
                    self.logger.info("epoch_global: {}, step_global: {}, step: {}".format(epochs_i, global_steps, idx))
                    self.logger.info("best_report\n" + best_report)
                    self.logger.info("current_mertics: {}".format(res))
                    # idx_score = res.get("micro", {}).get("f1", 0)  # "macro", "micro", "weighted"
                    for k,v in res.items():  # tensorboard日志, 其中抽取中文、数字和英文, 避免一些不支持的符号, 比如说 /\|等特殊字符
                        if type(v) == dict:  # 空格和一些特殊字符tensorboardx.add_scalar不支持
                            k = chinese_extract_extend(k)
                            k = k.replace(" ", "")
                            for k_2, v_2 in v.items():
                                tensorboardx_witer.add_scalar(k + "/" + k_2, v_2, global_steps)
                        elif type(v) == float:
                            tensorboardx_witer.add_scalar(k, v, global_steps)
                            tensorboardx_witer.add_scalar("lr", scheduler.get_lr()[-1], global_steps)
                    self.model.train()  # 预测时候的, 回转回来
                    save_best_mertics_key = self.config.save_best_mertics_key  # 模型存储的判别指标
                    abmk_1 = save_best_mertics_key[0]  # like "micro_avg"
                    abmk_2 = save_best_mertics_key[1]  # like "f1-score"
                    if res.get(abmk_1, {}).get(abmk_2, 0) > best_mertics.get(abmk_1, {}).get(abmk_2, 0):  # 只保留最优的指标
                        epochs_store.append((epochs_i, idx))
                        res["total"] = {"epochs": epochs_i, "global_steps": global_steps, "step_current": idx}
                        best_mertics = res
                        best_report = report
                        self.save_model()
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

