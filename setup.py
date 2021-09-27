# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/01 10:17
# @author   :Mo
# @function :setup of Pytorch-NLU


from pytorch_nlu.version import __version__
from setuptools import find_packages, setup
import codecs


# Package meta-data.
NAME = "Pytorch-NLU"
DESCRIPTION = "Pytorch-NLU"
URL = "https://github.com/yongzhuo/Pytorch-NLU"
EMAIL = "1903865025@qq.com"
AUTHOR = "yongzhuo"
LICENSE = "Apache"

with codecs.open("README.md", "r", "utf-8") as reader:
    long_description = reader.read()
with codecs.open("requirements.txt", "r", "utf-8") as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))

setup(name=NAME,
        version=__version__,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        packages=find_packages(),
        install_requires=install_requires,
        package_data={"pytorch_nlu": ["*.*", "corpus/*",
                                      "pytorch_textclassification/*"
                                      "pytorch_sequencelabeling/*",
                                      "corpus/text_classification/*",
                                      "corpus/sequence_labeling/*",
                                      "corpus/text_classification/school/*",
                                      "corpus/text_classification/tnews/*",
                                      "corpus/sequence_labeling/ner_china_people_daily_1998_conll/*",
                                      "corpus/sequence_labeling/ner_china_people_daily_1998_span/*",]},
        license=LICENSE,
        classifiers=["License :: OSI Approved :: Apache License",
                     "Programming Language :: Python :: 3.4",
                     "Programming Language :: Python :: 3.5",
                     "Programming Language :: Python :: 3.6",
                     "Programming Language :: Python :: 3.7",
                     "Programming Language :: Python :: 3.8",
                     "Programming Language :: Python :: 3.9",
                     "Programming Language :: Python :: Implementation :: CPython",
                     "Programming Language :: Python :: Implementation :: PyPy"],)


if __name__ == "__main__":
    print("setup ok!")


# 打包与安装
# step:
#     打开cmd
#     到达安装目录
#     python setup.py build
#     python setup.py install

# or

# python setup.py bdist_wheel --universal
# twine upload dist/*

