快速入门
===========

目前，我们在ink.algorithm模块下包含了
中文分词( :code:`ChineseWordSegmentation` )、
命名实体识别( :code:`NamedEntityRecognition` )、
词性标注( :code:`PosTagging` )、
细粒度实体分类( :code:`Typing`)
等算法类，你可以实例化出对应的类对象，
并调用其 :code:`__call__(self, sents: list[str])` 函数，并获取对应的返回值。

各算法类的具体接口可以参考API部分的对应文档。

在加载模型前，可以修改包括device内的多项配置，详情可以参考 :ref:`config` 文档。

模型第一次使用时，会先下载对应模型的资源文件，该资源文件会保存在本地，之后使用时会直接使用而无需重新下载。

.. code-block:: python

    from ink.config.config import config
    from ink.algorithm import ChineseWordSegmentation, NamedEntityRecognition, PosTagging


    config.default_device = "cuda: 1"
    sents = ['', '']
    result_ner = NamedEntityRecognition().ner(sents)
    result_cws = ChineseWordSegmentation().cws(sents)
    results_pt = PosTagging()(sents)

