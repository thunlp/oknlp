词性标注
=============

.. code-block:: python

    import oknlp


    config = {'batch_size': 16, 'device': 'cuda:0'}
    pos_tagging = oknlp.postagging.get_by_name("bert", **config)
    sents = ['我爱北京天安门']
    result = pos_tagging(sents)
    # result == [[('我', 'PN'), ('爱', 'VV'), ('北京', 'NR'), ('天安门', 'NR')]]
