import unittest
import oknlp


class TestTasks(unittest.TestCase):
    def test_ner(self):
        sents = ["清华大学自然语言处理与社会人文计算实验室"]
        for ner in oknlp.ner.get_all():
            results = ner(sents)
            self.assertEqual(len(sents), len(results))
            for sent, result in zip(sents, results):
                sent_r = ''.join([sent[word['begin']:word['end'] + 1] for word in result])
                self.assertEqual(sent, sent_r)

    def test_cws(self):
        sents = ["清华大学自然语言处理与社会人文计算实验室"]
        for cws in oknlp.cws.get_all():
            results = cws(sents)
            self.assertEqual(len(sents), len(results))
            for sent, result in zip(sents, results):
                sent_r = ''.join(result)
                self.assertEqual(sent, sent_r)

    def test_pos_tagging(self):
        sents = ["清华大学自然语言处理与社会人文计算实验室"]
        for pos_tagging in oknlp.postagging.get_all():
            results = pos_tagging(sents)
            self.assertEqual(len(sents), len(results))
            for sent, result in zip(sents, results):
                sent_r = ''.join([word for (word, tag) in result])
                self.assertEqual(sent, sent_r)

    def test_typing(self):
        sents = [("3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", (30, 33)),
                 ("张淑芳老人记得照片是在工人文化宫照的，而且还是在一次跳完集体舞后拍摄的。但摄影师是谁，照片背后的字是谁写的，已经找寻不到答案了。", (22, 24))]
        for typing in oknlp.typing.get_all():
            results = typing(sents)
            self.assertEqual(len(sents), len(results))
            for result in results:
                for entity_score in result:
                    (entity, score) = entity_score
                    self.assertIsInstance(entity, str)
                    self.assertIsInstance(score, float)
