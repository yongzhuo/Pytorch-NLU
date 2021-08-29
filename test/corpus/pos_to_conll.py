# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/8/24 23:09
# @author  : Mo
# @function: transform conll to span, 将CONLL格式的数据转化为MYZ格式{pos:[1,3]}


import logging


def txt_write(lines, path: str, model: str = "w", encoding: str = "utf-8"):
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
def save_json(lines, path, encoding: str = "utf-8", indent: int = 4):
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
def transform_span_to_conll(sent, label, sl_ctype):
    """将span格式数据(pos, SPAN)转化为CONLL的形式
    transform span to conll
    Args:
        label     : List<dcit>, span-pos,  eg. [{"type":"city", "ent":"沪", "pos":[2:3]}]
        sent      : str,  sent of one sample,  eg. "macropodus是叉尾斗鱼"
        sl_ctype  : str,  type of corpus, 数据格式sl-type,  eg. "BIO", "BMES", "BIOES" 
    Returns:
        res       : List, eg. [("鱼", "O")]
    """
    label_str = ["O"] * len(sent)
    for i, yi in enumerate(label):
        yi_pos = yi.get("pos", [0, 1])
        yi_type = yi.get("type", "")
        # yi_e = yi.get("ent", "")
        yi_pos_0 = yi_pos[0]
        yi_pos_1 = yi_pos[1]
        # 截取的最大长度, 防止溢出
        if yi_pos_1 >= len(sent):
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
    res = []
    for i in range(len(label_str)):
        res.append((sent[i], label_str[i]))
    return res
def read_corpus(corpus_path, encoding="utf-8", keys=["text", "label"]):
    """读取MYZ类型数据
    read corpus for sequence-labeling
    Args:
        corpus_path: String, path/origin text,  eg. "ner.conll"
    Returns:
        data: List<dict>,  [{...}]
    """
    with open(corpus_path, "r", encoding=encoding) as fo:
        xys = []
        count = 0
        for line in fo:
            count += 1
            # if count > 32:
            #     break
            if not line:
                continue
            # 最初想定义可配置化, 但是后期实验较多, 还是设置成一般形式, 可自己定义
            line_json = json.loads(line.strip())
            x, y = line_json.get(keys[0], ""), line_json.get(keys[1], [])
            xys.append((x, y))
        fo.close()
    return xys


if __name__ == '__main__':
    import json
    import sys
    import os
    path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(path_root)
    print(path_root)
    path = path_root + "/corpus/sequence_labeling/ner_china_people_daily_1998_span/"
    for t in ["train", "dev", "test"]:
        t = t + ".span"
        data = read_corpus(path + t)
        res = []
        for d in data:
            label = transform_span_to_conll(d[0], d[1], sl_ctype="BIO")
            label_strs = [li[0] + " " + li[1] + "\n" for li in label] + ["\n"]
            res += label_strs
        txt_write(res, path + t.replace(".span", ".conll"))

        ee = 0

    # transform span to conll, 将SPAN格式{pos:[1,3]}数据转化为CONLL格式

