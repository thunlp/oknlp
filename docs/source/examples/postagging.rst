PosTagging
===========

.. code-block:: python

    from ink.config.config import config
    from ink.algorithm import PosTagging


    config.default_device = "cuda: 1"
    pos_tagging = PosTagging()
    sents = ['我爱北京天安门']
    result = pos_tagging(sents)
