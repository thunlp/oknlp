NER
===========

.. code-block:: python

    from ink.config.config import config
    from ink.algorithm import NamedEntityRecognition


    config.default_device = "cuda: 1"
    ner = NamedEntityRecognition()
    sents = ['我爱北京天安门']
    result = ner(sents)
