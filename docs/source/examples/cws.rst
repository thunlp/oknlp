中文分词
=============

.. code-block:: python

    import oknlp


    cws = oknlp.cws.get_by_name(name="bert")
    sents = ['我爱北京天安门'] # 输入为list[str]
    result = cws(sents) # 输出为list
    # [list[str]] result == [['我', '爱', '北京', '天安门']]
