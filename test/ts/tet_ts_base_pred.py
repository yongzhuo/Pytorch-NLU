# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/25 9:30
# @author  : Mo
# @function: predict model, 预测模块


# 适配linux
from argparse import Namespace
import logging as logger
import traceback
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
path_sys = os.path.join(path_root, "pytorch_nlu", "pytorch_textsummary")
sys.path.append(path_sys)
print(path_root)
print(path_sys)

from tsPredict import TextSummaryPredict
from tsTools import load_json
from tsOffice import Office
from tsData import DataSet


if __name__ == "__main__":

    path_config = "../output/text_summary/model_BERT/tc.config"

    tcp = TextSummaryPredict(path_config)
    texts = [{'text': ['平乐县', '古称昭州', '隶属于广西壮族自治区桂林市', '位于广西东北部', '桂林市东南部', '东临钟山县',
               '南接昭平', '西北毗邻阳朔', '北连恭城', '总面积1919.34平方公里。']},
             {'text': ['平乐县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等', '平乐县为漓江分界点',
                  '平乐以北称漓江', '以南称桂江', '是著名的大桂林旅游区之一。']},
             {'text': ['印岭玲珑', '昭水晶莹', '环绕我平中。青年的乐园', '多士受陶熔。生活自觉自治', '学习自发自动。五育并重',
                  '手脑并用。迎接新潮流', '建设新平中']},
             {'text': ['桂林山水甲天下', '阳朔山水甲桂林']}]
    res = tcp.predict(texts, logits_type="sigmoid")
    print(str(res).encode("utf-8", "ignore").decode("utf-8", "ignore"))

