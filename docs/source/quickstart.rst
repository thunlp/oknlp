快速入门
===========

目前，我们在oknlp.algorithm模块下包含了
中文分词( :code:`cws` )、
命名实体识别( :code:`ner` )、
词性标注( :code:`postagging` )、
细粒度实体分类( :code:`typing`)
等算法模块，模块中包含使用不同算法的具体算法类，你可以实例化出对应的类对象，
并调用其 :code:`__call__(self, sents: list)` 函数，获取对应的输出。

模型第一次使用时，会先下载对应模型的资源文件，该资源文件会保存在本地，之后使用时会直接使用而无需重新下载。

对于不同模块的具体举例，可以参考Examples部分的文档。由于可能存在模型更新，代码运行后的实际结果可能与样例不同，这里仅体现接口格式。

对于中文分词模块，一个使用示例如下：

.. code-block:: python

    import oknlp


    # 设置kwargs参数传入算法类
    config = {'batch_size': 16, 'device': 'cuda:0'}
    # 调用oknlp包对应模块下的get_by_name(name)函数获取对应的算法类
    cws = oknlp.cws.get_by_name(name="bert", **config)
    # 输入对应格式的list
    sents = ['我爱北京天安门']
    result = cws(sents)
    # 输出list result == [['我', '爱', '北京', '天安门']]

