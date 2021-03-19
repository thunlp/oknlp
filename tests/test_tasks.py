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
        # output (for example):
        # [
        #  [['B-P', 'E-P', 'S-DT', ...], [((0, 2), 'P'), ((2, 3), 'DT'), ...]],
        #  [['B-JJ', 'E-JJ', 'B-NN', 'E-NN', ...], [((0, 2), 'JJ'), ((2, 4), 'NN'), ...]]
        # ]
        from ink.algorithm import PosTagging
        pos_tagging = PosTagging()
        results = pos_tagging(sents)
        self.assertEqual(len(sents), len(results))
        for idx, result in enumerate(results):
            self.assertEqual(len(sents[idx]), len(result[0]))
            last_end = 0
            for (begin, end), _ in result[1]:
                self.assertEqual(last_end, begin)
                last_end = end
            self.assertEqual(len(result[0]), last_end)

    def test_typing(self):
        from ink.algorithm import Typing
        typing = Typing()
        results = typing([["3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", [30, 33]]])
        for result in results:
            for entity_score in result:
                self.assertEqual(len(entity_score), 2)
                (entity, score) = entity_score
                self.assertIsInstance(entity, str)
                self.assertIsInstance(score, float)
