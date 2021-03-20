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
    # result == [[(‘object’, 0.35983458161354065), (‘event’, 0.8602959513664246), (‘attack’, 0.12778696417808533), (‘disease’, 0.2171688675880432)]]
