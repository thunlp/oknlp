Typing
=======

.. code-block:: python

    from ink.config.config import config
    from ink.algorithm import Typing


    config.default_device = "cuda: 1"
    typing = Typing()
    sents = [
        ["3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", [30, 33]],
    ]
    results = typing(sents)
