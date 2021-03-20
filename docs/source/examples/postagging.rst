PosTagging
===========

.. code-block:: python

    from ink.config.config import config
    from ink.algorithm import PosTagging


    config.default_device = "cuda: 1"
    pos_tagging = PosTagging()
    sents = ['我爱北京天安门']
    result = pos_tagging(sents)
    # result == [[('我', 'PN'), ('爱', 'VV'), ('北京', 'NR'), ('天安门', 'NR')]]
