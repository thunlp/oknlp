中文分词
=============

.. code-block:: python

    from ink.config.config import config
    from ink.algorithm.cws import get_cws


    config.default_device = "cuda: 1"
    cws = get_cws(name="bert")
    sents = ['我爱北京天安门']
    result = cws(sents)
    # result == ['我 爱 北京 天安门']
