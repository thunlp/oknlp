命名实体识别
========================

.. code-block:: python

    import oknlp


    ner = oknlp.ner.get_by_name(name="bert")
    sents = ['我爱北京天安门']  # 输入为list[str]
    result = ner(sents) # 输出为list[dict]
    # result == [[{'begin': 2, 'type': 'LOC', 'end': 3}, {'begin': 4, 'type': 'LOC', 'end': 6}]]
