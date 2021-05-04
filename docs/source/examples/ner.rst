命名实体识别
========================

.. code-block:: python

    import oknlp


    ner = oknlp.ner.get_by_name(name="bert")
    ner.to("cuda:1")
    sents = ['我爱北京天安门']
    result = ner(sents)
    # result == [[{'begin': 2, 'type': 'LOC', 'end': 3}, {'begin': 4, 'type': 'LOC', 'end': 6}]]
