CWS
===========

.. code-block:: python

    from ink.config.config import config
    from ink.algorithm import ChineseWordSegmentation


    config.default_device = "cuda: 1"
    cws = ChineseWordSegmentation()
    sents = ['我爱北京天安门']
    result = cws(sents)
    # result == ['我 爱 北京 天安门']
