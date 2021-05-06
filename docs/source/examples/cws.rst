中文分词
=============

.. code-block:: python

    import oknlp


    cws = oknlp.cws.get_by_name(name="bert")
    cws.to("cuda:1")
    sents = ['我爱北京天安门']
    result = cws(sents)
    # result == [['我', '爱', '北京', '天安门']]
