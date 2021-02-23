import unittest


sents = ['我爱北京天安门']
class TestTasks(unittest.TestCase):
    def test_ner(self):
        from ink.algorithm import NamedEntityRecognition
        # only test input.shape[0] == output.shape[0] (which is batch_size)
        re1 = [[[{'begin': 2, 'type': 'LOC', 'end': 3}, {'begin': 4, 'type': 'LOC', 'end': 6}]]]
        result_ner = NamedEntityRecognition().ner(sents)
        self.assertTrue(result_ner,re1)

    def test_cws(self):
        from ink.algorithm import ChineseWordSegmentation
        re2 = ['我 爱 北 京 天 安门']
        result_cws = ChineseWordSegmentation().cws(sents)
        self.assertTrue(result_cws, re2)
