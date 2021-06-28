================
词性标注
================
这篇文档主要介绍了词性标注算法的输入、输出以及各算法支持的参数配置。

词性标注接口
=======================
.. autoclass:: oknlp.postagging.BasePosTagging()
    :members: __call__

词性标注实现
=======================

BERT
-----------------------
.. autoclass:: oknlp.postagging.BertPosTagging()
