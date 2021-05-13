词性标注
=============

.. code-block:: python

    import oknlp


    pos_tagging = oknlp.postagging.get_by_name("bert")
    pos_tagging.to("cuda:1")
    sents = ['我爱北京天安门']
    result = pos_tagging(sents)
    # result == [[('我', 'PN'), ('爱', 'VV'), ('北京', 'NR'), ('天安门', 'NR')]]
