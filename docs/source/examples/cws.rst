中文分词
=============

.. code-block:: python

    import oknlp


    config = {'batch_size': 16, 'device': 'cuda:0'}
    cws = oknlp.cws.get_by_name(name="bert", **config)
    sents = ['我爱北京天安门']
    result = cws(sents)
    # result == [['我', '爱', '北京', '天安门']]
