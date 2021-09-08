# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:35
# @author  : Mo
# @function: utils of Pytorch-SequenceLabeling


from collections import OrderedDict, defaultdict
from logging.handlers import RotatingFileHandler
from typing import Union, Dict, List, Any
import logging
import numpy as np
import time
import json
import re
import os


__all_ = ["mertics_report_sequence_labeling",
          "yongzhuo_confusion_matrix",
          "chinese_extract_extend",
          "is_total_chinese",
          "get_current_time",
          "delete_file",
          "get_logger"
          "save_json",
          "load_json",
          "txt_write",
          "txt_read",
          "dic_sort",
          "del_dir",
          ]


def mertics_report_sequence_labeling(y_true_dict: List, y_pred_dict: List, idx2label:Dict=None, rounded: int=4, use_draw:bool=True,
                                     xy_keys: List=["text", "label"], label_keys: List=["type", "ent", "pos"]):
    """ 序列标注-评估指标, 只支持list<json>格式输入的y_true_dict/y_pred_dict, position格式
    支持例如实体识别NER、中文分词CWS、槽位识别SF
    mertics report sequence labeling, 打印评估指标, 不支持CONLL的形式
    Args:
        y_true_dict:  List, 真实样本标签, eg.[{"text": "咱们桂林市是个好地方", "entities": [{"entity_type": "ORG", "start_pos": 2, "end_pos": 5, "word": "桂林市"}]}]
        y_pred_dict:  List, 预测样本标签, eg.[{"text": "咱们桂林市是个好地方", "entities": [{"entity_type": "ORG", "start_pos": 2, "end_pos": 4, "word": "桂林"}]}]
        rounded    :  int,  最后保留小数位
        use_draw   :  bool, 是否生成string格式的报告提供日志显示
        xy_keys    :  List, selected key of json in x and y
        label_keys :  List, selected key of json in label
    Returns:  tuple<float>, 多个返回值
        mertics:  dict, 包括单个类别/micro/macro/weighted的 prf
        report:   str,  评估指标打印格式, 同sklearn
        mcm:      List<int>, 混淆矩阵, shape=(n_labels+1, n_labels+1)
        idx:      List<int>, 预测有错误的未知
    """
    # assert len(y_true_dict) == len(y_pred_dict), "lengths of y_pred and y_true do not match."

    def transform_pos_to_conll(y, xy_keys=["text", "label"], label_keys=["type", "ent", "pos"]):
        """ 将位置信息转为conll(list)的形式
        Args:
            y         : dict, pos格式的原语料和标注语料
            xy_keys   : list, selected key of json in x and y
            label_keys: list, selected key of json in label
        Returns:    
            type_labels: list, 类别的conll形式
        """
        type_labels = ["O" for _ in y[xy_keys[0]]]
        labels = y[xy_keys[1]]
        for label in labels:
            label_type = label[label_keys[0]]
            pos_start = label[label_keys[2]][0]
            pos_end = label[label_keys[2]][1]
            for j in range(pos_start, pos_end):
                type_labels[j] = label_type
        return type_labels
    def calculate_precision_recall_f1score(count_true, count_pred, count_same):
        """ 计算精确率/召回率/F1得分
        Args:
            count_true:  int, 类别下的真实样本标签个数
            count_pred:  int, 类别下的预测样本标签个数
            count_same:  int, 类别下的正确样本标签个数
            eps       :  float, 设置一个eps防止为0
        Returns:      tuple<float>, 多个返回值， prf
        """
        precision = count_same / count_pred if count_pred else 0
        recall = count_same / count_true if count_true else 0
        f1 = (2 * count_same) / (count_true + count_pred) if (count_true + count_pred) else 0
        return precision, recall, f1
    def draw_mertics_dict(metrics, rounded=4, use_acc=False):
        """
        draw mertics, 绘画评估指标
        Args:
            metrics: dict, eg. {"游戏:132, "旅游":32}
            rounded: int, eg. 4
            use_acc: bool, add acc or not, eg.True
        Returns:
            report: str
        """
        # 打印头部
        labels = list(metrics.keys())
        target_names = [lab for lab in labels]  # [u"%s" % l for l in labels]
        name_width = max(len(str(cn)) for cn in target_names)
        width = max(name_width, rounded)
        headers = list(metrics[labels[0]].keys())
        if not use_acc and "acc" in headers: headers.remove("acc")
        head_fmt = u"{:>{width}} " + u" {:>9}" * len(headers)
        report = "\n\n" + head_fmt.format("", *headers, width=width)
        report += "\n\n"
        # 具体label的评估指标
        count = 0
        for lab in labels:
            if lab in ["macro_avg", "micro_avg", "weighted_avg"] and count == 0:
                report += "\n"
                count += 1
            row_fmt = u"{:>{width}} " + u" {:>9.{rounded}f}" * (len(headers) - 1) + u" {:>9.{support_rounded}f}\n"
            if use_acc:
                [p, r, f1, a, s] = [metrics[lab][hd] for hd in headers]
                report += row_fmt.format(lab, p, r, f1, a, s, width=width, rounded=rounded, support_rounded=0)
            else:
                [p, r, f1, s] = [metrics[lab][hd] for hd in headers]
                report += row_fmt.format(lab, p, r, f1, s, width=width, rounded=rounded, support_rounded=0)
        report += "\n"
        return report
    def draw_mcm(idx2label, mcm, rounded=0):
        """
        draw mcm, 绘画混淆矩阵
        Args:
            idx2label : dict, eg. {"游戏:0, "旅游":1}
            mcm       : list, eg. [[1,2], [2,1]]
        Returns:
            report: str
        """
        # 打印头部
        target_names = [idx2label[str(i)] for i in range(len(idx2label))]  # [u"%s" % l for l in labels]
        name_width = max(len(str(cn)) for cn in target_names)
        width = max(name_width, rounded)
        headers = target_names
        head_fmt = u"{:>{width}} " + u" {:>9}" * len(headers)
        report = "\n\n" + head_fmt.format("", *headers, width=width)
        report += "\n\n"
        # 具体label的评估指标
        for i in range(len(target_names)):
            lab = target_names[i]
            row_fmt = u"{:>{width}} " + u" {:>9.{rounded}f}" * len(headers) + " \n"
            args = [lab] + [int(m) for m in mcm[i]]
            report += row_fmt.format(*args, width=width, rounded=rounded)
        report += "\n"
        return report

    len_sent = len(y_true_dict)
    # key, 字典的主键key
    y_type_counter = {"y_true": defaultdict(int), "y_pred": defaultdict(int), "y_same": defaultdict(int)}
    key_label_type = label_keys[0]
    # key_label_ent = label_keys[1]
    key_label_pos = label_keys[2]
    key_laebl = xy_keys[1]
    # key_text = xy_keys[0]
    y_error_dict = []  # 预测错误的数据
    y_true_conlls, y_pred_conlls = [], []
    # 计算PRF指标
    for i in range(len_sent):
        y_true_i = y_true_dict[i].get(key_laebl, [])
        y_pred_i = y_pred_dict[i].get(key_laebl, [])
        y_true_conlls += transform_pos_to_conll(y_true_dict[i], xy_keys, label_keys)
        y_pred_conlls += transform_pos_to_conll(y_pred_dict[i], xy_keys, label_keys)
        # 存储 y_true的类型个数, 位置
        y_true_pos_counter = defaultdict(list)
        for yti in y_true_i:
            yti_type = yti.get(key_label_type, "")
            yti_pos = yti.get(key_label_pos, "")
            y_type_counter["y_true"][yti_type] += 1
            y_true_pos_counter[yti_type] += [yti_pos]
        # 存储 y_pred、y_match的类型个数, 位置
        flag = False
        for ypi in y_pred_i:
            ypi_type = ypi.get(key_label_type, "")
            ypi_pos = ypi.get(key_label_pos, "")
            y_type_counter["y_pred"][ypi_type] += 1
            # pos对应上的, 完全匹配, 即pos=[0, 1]全对才算是一个实体对
            if ypi_pos in y_true_pos_counter[ypi_type]:
                y_type_counter["y_same"][ypi_type] += 1
            else:
                flag = True
        if flag:  # 错误信息
            y_true_dict[i].update({key_laebl+"_pred": y_pred_i})
            y_error_dict.append(y_true_dict[i])

    # 按照类别样例多少进行排序, 为了好看一点, 数量级高的展示在前面
    if not idx2label:
        idx2label = {str(i):yx[0] for i,yx in enumerate(sorted(y_type_counter["y_true"].items(), key=lambda x:x[1], reverse=True))}
    entity_types = [idx2label[str(i)] for i in range(len(idx2label))]
    # 混淆矩阵
    cm, label_to_ind = yongzhuo_confusion_matrix(y_true_conlls, y_pred_conlls, labels=entity_types)
    # label-prf, 只取实体的, 是city而不是B-city
    entity_types = [et for et in entity_types if et in y_type_counter["y_true"].keys()]
    mertics_label = []
    for entity_type in entity_types:
        count_type_true = y_type_counter["y_true"].get(entity_type, 0)
        count_type_pred = y_type_counter["y_pred"].get(entity_type, 0)
        count_type_same = y_type_counter["y_same"].get(entity_type, 0)
        precision, recall, f1score = calculate_precision_recall_f1score(count_type_true, count_type_pred, count_type_same)
        mertics_label.append([precision, recall, f1score, count_type_true])

    if mertics_label and entity_types:
        # global-prf of "micro", "macro", "weighted"
        mertics_label = np.array(mertics_label)
        tps = [sum([y_type_counter[y_t][et] for et in entity_types]) for y_t in ["y_true", "y_pred","y_same"]]
        sum_precision, sum_recall, sum_f1score = calculate_precision_recall_f1score(tps[0], tps[1], tps[2])
        mertics_micro = [sum_precision, sum_recall, sum_f1score, tps[0]]
        mertics_weighted = (np.dot(mertics_label[:, -1:].T, mertics_label[:, :-1]) / tps[0]).sum(0).tolist()
        mertics_weighted += [tps[0]]
        mertics_macro = mertics_label.sum(0) / len(entity_types)
        mertics_macro = mertics_macro.tolist()
        mertics_macro[-1] = tps[0]
        mertics_list = mertics_label.tolist() + [mertics_micro] + [mertics_macro] + [mertics_weighted]
        # mertics_dict
        entity_types = entity_types + ["micro_avg", "macro_avg", "weighted_avg"]
        mertics_names = ["precision", "recall", "f1-score", "support"]
        mertics_dict = {}
        for i in range(len(entity_types)):
            et = entity_types[i]
            mertics_dict[et] = {}
            for j in range(len(mertics_list[i])):
                if mertics_names[j] == mertics_names[-1]:
                    mertics_dict[et][mertics_names[j]] = round(mertics_list[i][j], 0)
                else:
                    mertics_dict[et][mertics_names[j]] = round(mertics_list[i][j], rounded)
        # 绘制打印
        mertics_report = ""
        mcm_report = ""
        if use_draw:
            mertics_report = draw_mertics_dict(mertics_dict)
            mcm_report = draw_mcm(idx2label, cm.tolist())
        return mertics_dict, mertics_report, mcm_report, y_error_dict
    else:
        return {}, "", "", []


def yongzhuo_confusion_matrix(y_true, y_pred, labels=None, normalize=None):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    args:
        y_true : array-like of shape (n_samples,)， Ground truth (correct) target values.
        y_pred : array-like of shape (n_samples,)， Estimated targets as returned by a classifier.
        labels : array-like of shape (n_classes), default=None， List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    url: https://github.com/scikit-learn/scikit-learn
    Returns:
        cm     : array-like of shape (n_classes)
    Examples:
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    """

    def myz_coo_matrix(args, shape=None, dtype=None):
        """coo-matrix是最简单的稀疏矩阵存储方式，采用三元组(row, col, data, 即ijv format的形式)来存储矩阵中非零元素。
        在实际使用中，一般coo_matrix用来创建矩阵，因为coo_matrix无法对矩阵的元素进行增删改操作；
        创建成功之后可以转化成其他格式的稀疏矩阵（如csr_matrix、csc_matrix）进行转置、矩阵乘法等操作。
        myz_coo_matrix, coo_matrix of myz
        args:
            args : tuple, three input data of coo matrix, 输入稀疏数据, eg. ([1,2,3], ([4,5,6], [7,8,9]))
            shape: tuple, input data shape of coo matrix, 输入数据的形状, eg. (3,3)
            dtype: numpy.dtype, data-type of numpy, numpy工具的类型, eg. np.int64
        Returns:
            matrix: np.array
        """
        _data, (_row, _col) = args
        shape = shape
        dtype = dtype
        _data = _data
        _row = _row
        _col = _col
        matrix = np.zeros(shape=shape, dtype=dtype)
        for i in range(len(_col)):
            matrix[_row[i]][_col[i]] += _data[i]
        return matrix

    if not labels:
        labels = set(y_true)
    n_labels = len(labels)
    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_true) == list:
        y_pred = np.array(y_pred)
    sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    label_to_ind = {y: x for x, y in enumerate(labels)}
    # convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    # also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]
    # Choose the accumulator dtype to always have high precision
    if sample_weight.dtype.kind in {"i", "u", "b"}:
        dtype = np.int64
    else:
        dtype = np.float64

    cm = myz_coo_matrix((sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels), dtype=dtype)

    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)
    return cm, label_to_ind


def save_json(lines: Union[List, Dict], path: str, encoding: str = "utf-8", indent: int = 4):
    """
    Write Line of List<json> to file
    Args:
        lines: lines of list[str] which need save
        path: path of save file, such as "json.txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    with open(path, "w", encoding=encoding) as fj:
        fj.write(json.dumps(lines, ensure_ascii=False, indent=indent))
    fj.close()


def txt_write(lines: List[str], path: str, model: str = "w", encoding: str = "utf-8"):
    """
    Write Line of list to file
    Args:
        lines: lines of list<str> which need save
        path: path of save file, such as "txt"
        model: type of write, such as "w", "a+"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    try:
        file = open(path, model, encoding=encoding)
        file.writelines(lines)
        file.close()
    except Exception as e:
        logging.info(str(e))


def load_json(path: str, encoding: str = "utf-8") -> Union[List, Any]:
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json


def txt_read(path: str, encoding: str = "utf-8") -> List[str]:
    """
    Read Line of list form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        dict of word2vec, eg. {"macadam":[...]}
    """

    lines = []
    try:
        file = open(path, "r", encoding=encoding)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logging.info(str(e))
    finally:
        return lines


def get_pos_from_span(logits_start: List, logits_end: List, i2l: Dict, use_index=False) -> List:
    """从span模型的输出中重构标注, 即获取未知信息---position
    span analysis for sequence-labeling
    Args:
        logits_start: List<list>, span-output of 
        logits_end: List<list>, string form of return time
        idx2label: dict, string form of return time
        use_index: bool, True is from label, False is from logits
    Returns:
        pos: List, eg. [(人名, 0, 2)]
    """
    if not use_index:
        pos_start = [x.index(max(x)) for x in logits_start]
        pos_end = [x.index(max(x)) for x in logits_end]
    else:
        pos_start = logits_start
        pos_end = logits_end
    pos = []
    for i, ps in enumerate(pos_start):
        ps = int(ps)
        if ps == 0:  # 从左往右, 从0开始
            continue
        for j, pe in enumerate(pos_end[i:]):
            pe = int(pe)
            if ps == pe:
                pos.append((i2l[str(ps)], i, i + j))
                # if i==j:
                #     pos.append((i2l[str(ps)], i, i+j))
                # for ij in range(i, j):
                #     pos.append((i2l[str(ps)], i, i+j))
                break
    return pos


def transform_span_to_conll(y, label_id, l2i_conll, sl_ctype):
    """将span格式数据(pos, SPAN)转化为CONLL的形式
    transform span to conll
    Args:
        y         : List<dcit>, span-pos,  eg. [{"type":"city", "ent":"沪", "pos":[2:3]}]
        label_id  : List<int>,  label of one sample,  eg. [0, 0, 0, 0]
        l2i       : Dict, dict of label,  eg. {"O":0, "ORG":1} or {"O":0, "B-ORG":1, "I-ORG":2}
        l2i_conll : Dict, dict of label extend,  eg. {"O":0, "B-ORG":1, "I-ORG":2}
        sl_ctype  : str, type of corpus, 数据格式sl-type,  eg. "BIO", "BMES", "BIOES" 
    Returns:
        reault: List, eg. [0, 0, 1, 0]
    """
    label_str = ["O"] * len(label_id)
    for i, yi in enumerate(y):
        yi_pos = yi.get("pos", [0, 1])
        yi_type = yi.get("type", "")
        # yi_e = yi.get("ent", "")
        yi_pos_0 = yi_pos[0]
        yi_pos_1 = yi_pos[1]
        # 截取的最大长度, 防止溢出
        if yi_pos_1 >= len(label_id):
            break
        if sl_ctype in ["BIO", "OIB"]:
            for id in range(yi_pos[1] - yi_pos[0]):
                label_str[yi_pos_0 + id] = "I-" + yi_type
            label_str[yi_pos_1] = "I-" + yi_type
            label_str[yi_pos_0] = "B-" + yi_type
        elif sl_ctype in ["BMES"]:  # 专门用于CWS分词标注等
            label_str[yi_pos_1] = "E-" + yi_type
            label_str[yi_pos_0] = "B-" + yi_type
            for id in range(yi_pos[1] - yi_pos[0]):
                label_str[yi_pos_0 + id] = "M-" + yi_type
            if yi_pos_0==yi_pos_1:
                label_str[yi_pos_0] = "S-" + yi_type
        elif sl_ctype in ["BIOES"]:
            label_str[yi_pos_1] = "E-" + yi_type
            label_str[yi_pos_0] = "B-" + yi_type
            for id in range(yi_pos[1] - yi_pos[0]):
                label_str[yi_pos_0 + id] = "I-" + yi_type
            if yi_pos_0 == yi_pos_1:
                label_str[yi_pos_0] = "S-" + yi_type
    label_id = [l2i_conll[s] for s in label_str]
    return label_id


def get_pos_from_common(words0, tag1, sep="-"):
    """从common模型的输出中重构标注, 即获取未知信息---position
    common analysis for sequence-labeling
    Args:
        words0: String, origin text,  eg. "沪是上海"
        tag1  : List, common-output of labels,  eg. ["S-city", "O", "B-city", "I-city"]
        sep   : String, split in tag, eg. "-"、"_"
    Returns:
        reault: List, eg. [{"type":"city", "ent":"沪", "pos":[2:4]}]
    """
    res = []
    ws = ""
    start_pos_1 = 0
    end_pos_1 = 0
    sentence = ""
    types = ""
    for i in range(len(tag1)):
        if tag1[i].startswith("S" + sep):
            ws += words0[i]
            start_pos_1 = i
            end_pos_1 = i
            sentence += words0[i]
            types = tag1[i][2:]
            res.append([ws, start_pos_1, end_pos_1, types])
            ws = ""
            types = ""

        if tag1[i].startswith("B" + sep):
            if len(ws) > 0:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""
            if len(ws) == 0:
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                sentence += words0[i]
                types = tag1[i][2:]

        elif tag1[i].startswith("I" + sep):
            if len(ws) > 0 and types == tag1[i][2:]:
                ws += words0[i]
                sentence += words0[i]
                end_pos_1 = i

            elif len(ws) > 0 and types != tag1[i][2:]:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            if len(ws) == 0:
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                sentence += words0[i]
                types = tag1[i][2:]

        elif tag1[i].startswith("M" + sep):
            if len(ws) > 0 and types == tag1[i][2:]:
                ws += words0[i]
                sentence += words0[i]
                end_pos_1 = i

            elif len(ws) > 0 and types != tag1[i][2:]:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            if len(ws) == 0:
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                sentence += words0[i]
                types = tag1[i][2:]

        elif tag1[i].startswith("E" + sep):
            if len(ws) > 0 and types == tag1[i][2:]:
                ws += words0[i]
                sentence += words0[i]
                end_pos_1 = i
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            if len(ws) > 0 and types != tag1[i][2:]:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                sentence += words0[i]
                types = tag1[i][2:]
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

        elif tag1[i] == "O":

            if len(ws) > 0:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            sentence += words0[i]

        if i == len(tag1) - 1 and len(ws) > 0:
            res.append([ws, start_pos_1, end_pos_1, types])
            ws = ""
            types = ""
    reault = []
    for r in res:
        entity_dict = {}
        entity_dict["type"] = r[3]
        entity_dict["ent"] = r[0]
        entity_dict["pos"] = [r[1], r[2]]
        reault.append(entity_dict)
    return reault


def get_logger(log_dir: str, back_count: int = 32, logger_name: str = "pytorch-nlp"):
    """
    get_current_time from time
    Args:
        log_dir: str, log dir of path
        back_count: int, max file-name
        logger_name: str
    Returns:
        logger: class
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # 日志文件名,为启动时的日期
    log_file_name = time.strftime("{}-%Y-%m-%d".format(logger_name), time.localtime(time.time())) + ".log"
    log_name_day = os.path.join(log_dir, log_file_name)
    logger_level = logging.INFO
    # log目录地址
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # 全局日志格式
    logging.basicConfig(format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                        level=logger_level)
    # 定义一个日志记录器
    logger = logging.getLogger("pytorch-nlp")
    # 文件输出, 定义一个RotatingFileHandler，最多备份32个日志文件，每个日志文件最大32K
    fHandler = RotatingFileHandler(log_name_day, maxBytes=back_count * 1024 * 1024, backupCount=back_count,
                                   encoding="utf-8")
    fHandler.setLevel(logger_level)
    logger.addHandler(fHandler)

    return logger


def get_current_time(time_form: str = "%Y%m%d%H%M%S"):
    """获取当前时间戳
    get_current_time from time
    Args:
        time_form: str, string form of return time
    Returns:
        time_current: str
    """
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    time_current = time.strftime(time_form, time_local)
    return time_current


def chinese_extract_extend(text: str) -> str:
    """
    只提取出中文、字母和数字，这里用于tensorboardX的label带顿号、空格等特殊字的时候，因为这些不支持
    Args:
        text: str, input of sentence
    Returns:
        chinese_extract: str
    """
    chinese_extract = "".join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@. ])", text))
    return chinese_extract


def is_total_chinese(text: str) -> bool:
    """
    judge is total chinese or not, 判断是不是全是中文
    Args:
        text: str, eg. "macadam, 碎石路"
    Returns:
        bool, True or False
    """
    for word in text:
        if not "\u4e00" <= word <= "\u9fa5":
            return False
    return True


def is_total_number(text: str) -> bool:
    """
    judge is total chinese or not, 判断是不是全是数字
    Args:
        text: str, eg. "macadam, 碎石路"
    Returns:
        bool, True or False
    """
    for word in text:
        if word not in "0123456789.%":
            return False
    return True


def dic_sort(dic: dict) -> OrderedDict:
    """
    sort dict by values, 给字典排序(依据值大小)
    Args:
        dic: dict, eg. {"游戏:132, "旅游":32}
    Returns:
        OrderedDict
    """
    in_dict_sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    return OrderedDict(in_dict_sort)


def del_dir(path_dir: str):
    """
    Delete model files in the directory, eg. h5/json/pb 
    Args:
        path_dir: path of directory, where many files in the directory
    Returns:
        None
    """
    for i in os.listdir(path_dir):
        # 取文件或者目录的绝对路径
        path_children = os.path.join(path_dir, i)
        if os.path.isfile(path_children):
            if path_children.endswith(".h5") or path_children.endswith(".json") or path_children.endswith(".pb"):
                os.remove(path_children)
        else:  # 递归, 删除目录下的所有文件
            del_dir(path_children)


if __name__ == '__main__':
    # unit tet
    y_true = ["O", "I", "O", "B"]
    y_pred = ["I", "I", "O", "B"]

    confusion_matrix = yongzhuo_confusion_matrix(y_true, y_pred)
    print(confusion_matrix)

    print(get_current_time())

    # sequence-labeling
    words0 = "沪是上海"
    tags1 = ["S-city", "O", "B-city", "I-city"]
    print(get_pos_from_common(words0, tags1))

