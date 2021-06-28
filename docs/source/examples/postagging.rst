词性标注
=============

.. code-block:: python

    import oknlp


    pos_tagging = oknlp.postagging.get_by_name("bert")
    sents = ['我爱北京天安门']  # 输入为list[str]
    result = pos_tagging(sents) # 输出为list[list[tuple]]
    # result == [[('我', 'PN'), ('爱', 'VV'), ('北京', 'NR'), ('天安门', 'NR')]]
