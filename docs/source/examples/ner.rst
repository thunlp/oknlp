命名实体识别
========================

.. code-block:: python

    import oknlp


    config = {'batch_size': 16, 'device': 'cuda:0'}
    ner = oknlp.ner.get_by_name(name="bert", **config)
    sents = ['我爱北京天安门']
    result = ner(sents)
    # result == [[{'begin': 2, 'type': 'LOC', 'end': 3}, {'begin': 4, 'type': 'LOC', 'end': 6}]]
