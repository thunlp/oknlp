中文分词
=============

.. code-block:: python

    import ink


    ink.config.default_device = "cuda: 1"
    cws = ink.cws.get_cws(name="bert")
    sents = ['我爱北京天安门']
    result = cws(sents)
    # result == [['我', '爱', '北京', '天安门']]
