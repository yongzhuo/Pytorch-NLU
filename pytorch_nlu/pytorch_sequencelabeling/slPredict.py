# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/25 19:30
# @author  : Mo
# @function: predict model, 预测模块


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(path_root)
from tcConfig import model_config
os.environ["CUDA_VISIBLE_DEVICES"] = model_config.get("CUDA_VISIBLE_DEVICES", "0")
from slConfig import _SL_MODEL_SOFTMAX, _SL_MODEL_GRID, _SL_MODEL_SPAN, _SL_MODEL_CRF
from slConfig import _SL_DATA_CONLL, _SL_DATA_SPAN
from slTools import get_logger, load_json
from slOffice import Office
from slData import Corpus

from argparse import Namespace


class SequenceLabelingPredict:
    def __init__(self, path_config):
        """ 初始化 """
        self.load_config(path_config)
        self.load_model()

    def load_config(self, path_config):
        """ 加载超参数  """
        config = load_json(path_config)
        self.config = Namespace(**config)
        self.logger = get_logger(self.config.model_save_path)
        self.l2i, self.i2l = self.config.l2i, self.config.i2l
        # 数据预处理 类
        self.corpus = Corpus(self.config, self.logger)

    def load_model(self):
        """ 加载模型  """
        self.office = Office(config=self.config, logger=self.logger)
        self.office.load_model()

    def process(self, texts):
        """ 数据预处理, process """
        # token 转 idx, 训练集/验证集
        datas_xy, _ = self.corpus.read_texts_from_json(texts, keys=self.config.xy_keys_predict)
        if self.config.task_type.upper() in [_SL_MODEL_SPAN]:
            sl_preprocess = self.corpus.preprocess_span
        elif self.config.task_type.upper() in [_SL_MODEL_GRID]:
            sl_preprocess = self.corpus.preprocess_grid
        else:
            sl_preprocess = self.corpus.preprocess_common
        dataset = sl_preprocess(datas_xy, self.config.l2i, l2i_conll=self.config.l2i_conll, sl_ctype=self.config.sl_ctype, max_len=self.config.max_len)
        return dataset

    def predict(self, texts):
        """ 预测 """
        dataset = self.process(texts)
        res = self.office.predict(dataset)
        return res


if __name__ == "__main__":

    path_config = "../output/sequence_labeling/model_ERNIE/sl.config"
    tcp = SequenceLabelingPredict(path_config)
    texts = [{"text": "平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919.34平方公里。"},
             {"text": "平乐县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点，平乐以北称漓江，以南称桂江，是著名的大桂林旅游区之一。"},
             {"text": "印岭玲珑，昭水晶莹，环绕我平中。青年的乐园，多士受陶熔。生活自觉自治，学习自发自动。五育并重，手脑并用。迎接新潮流，建设新平中"},
             {"text": "桂林山水甲天下, 阳朔山水甲桂林"},
             ]
    res = tcp.predict(texts)
    print(res)
    while True:
        print("请输入:")
        question = input()
        res = tcp.predict([{"text": question}])
        print(res)

