===========
快速入门
===========

在这篇文章中，我们会详细的讲解一个最简单的用例，让您快速的掌握OKNLP的基础使用方法。

基础使用方法
======================

要使用OKNLP，主要需要以下三个步骤：

1. 引入oknlp工具包
2. 创建并加载模型
3. 调用模型

接下来的代码将会逐步的进行演示。

引入工具包
-------------------------

你可以像引入其它Python库一样引入OKNLP工具包，它主要依赖于`onnxruntime`库。

.. code-block:: python

    import oknlp


如果在引入的过程中出现了错误，例如提示 ``OSError: libcublas.so.10: cannot open shared object file: No such file or directory`` ，请根据 :doc:`安装方法<installation>` 文档中的指示，检查你的安装步骤是否正确、是否已经安装好了工具包所必须的依赖。

在引入OKNLP工具包的过程中，还有可能会出现以下的 **警告** ，不过这是正常的情况，并不会影响到程序的正常运行，也不会对运行结果产生影响。


>>> import oknlp
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.


创建并加载模型
---------------------------

在使用OKNLP的算法之前，我们需要先创建并加载模型。在这里，我们以中文分词为例子进行演示。

.. code-block:: python

    model = oknlp.cws.get_by_name("bert")

在第一次加载模型时，OKNLP工具包会先下载模型的参数，并显示一个数据下载的进度条。而在之后的使用过程中，模型加载将直接使用之前下载好的参数。

下载好的模型参数会保存在用户目录下的``.oknlp``文件夹中。

调用模型
--------------------

在模型加载完成后，可以像调用普通函数一样来调用模型。OKNLP的模型通常输入都是一个 Python 的 `list` 对象。例如在分词任务中，输入就是一个句子的 `list` 。

和其它传统的分词工具不同（例如 **jieba** ），OKNLP使用了神经网络模型，因此一次输入多个句子可以得到更好的运行吞吐量。


>>> result = model([
...   "我爱北京天安门",
...   "天安门上太阳升"
... ])
>>> result
[['我', '爱', '北京', '天安门'], ['天安门', '上', '太阳', '升']]

在目前的版本中，OKNLP的接口是同步阻塞的，而异步的调用接口将会在之后的版本中提供。

完整示例代码
=========================

.. code-block:: python
    :linenos:

    # 引入工具包
    import oknlp

    # 加载模型
    model = oknlp.cws.get_by_name("bert")

    result = model([
        "我爱北京天安门",
        "天安门上太阳升"
    ])
    print(result)

**输出结果**

.. topic:: output
    :class: style-demo

    .. code-block:: python

        [['我', '爱', '北京', '天安门'], ['天安门', '上', '太阳', '升']]


线程 / 进程安全说明
===========================

OKNLP工具包提供的模型接口都是线程/进程安全的，这意味着你可以在多个线程/进程中同时调用同一个模型，并获得正确的结果。

在通常情况下，我们更建议您使用多线程来调用模型，这样可以获得更高的吞吐量和更小的系统资源消耗。