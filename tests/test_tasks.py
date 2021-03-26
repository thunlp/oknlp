import unittest


sents = ['我爱北京天安门']
class TestTasks(unittest.TestCase):

    """ FIXME: 现有模型由于版本原因需要更换
    def test_ner(self):
        from ink.algorithm import NamedEntityRecognition
        # only test input.shape[0] == output.shape[0] (which is batch_size)
        re1 = [[[{'begin': 2, 'type': 'LOC', 'end': 3}, {'begin': 4, 'type': 'LOC', 'end': 6}]]]
        result_ner = NamedEntityRecognition().ner(sents)
        self.assertEqual(result_ner,re1)

    def test_cws(self):
        from ink.algorithm import ChineseWordSegmentation
        re2 = ['我 爱 北 京 天 安门']
        result_cws = ChineseWordSegmentation().cws(sents)
        self.assertEqual(result_cws, re2)
    """

    def test_pos_tagging(self):
        from ink.algorithm.postagging import get_all_pos_tagging
        sents = ["清华大学自然语言处理与社会人文计算实验室"]
        for pos_tagging in get_all_pos_tagging():
            results = pos_tagging(sents)
            self.assertEqual(len(sents), len(results))
            for sent, result in zip(sents, results):
                sent_r = ''.join([word for (word, tag) in result])
                self.assertEqual(sent, sent_r)

    def test_typing(self):
        from ink.algorithm.typing import get_all_typing
        sents = [("3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", [30, 33]),
                 ("张淑芳老人记得照片是在工人文化宫照的，而且还是在一次跳完集体舞后拍摄的。但摄影师是谁，照片背后的字是谁写的，已经找寻不到答案了。", [22, 24])]
        for typing in get_all_typing():
            results = typing(sents)
            self.assertEqual(len(sents), len(results))
            for result in results:
                for entity_score in result:
                    (entity, score) = entity_score
                    self.assertIsInstance(entity, str)
                    self.assertIsInstance(score, float)
