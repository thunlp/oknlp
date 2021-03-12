# coding=UTF-8
import lacthu
from ink.data import load
class CWSbyTHULAC:
    def __init__(self):
       model_path = load('models')
       self.lac = lacthu.THUlac(model_path) 
    def lac_cws(self,sents):
        result = []
        for sent in sents:
            tmp = self.lac.THUlac_cws(sent)
            result.append((tmp.decode('utf-8')))
        return result
if __name__ =='__main__':

    lac  = CWSbyTHULAC()
    res = lac.lac_cws(['我爱北京天安门','天安门上太阳升'])
    print(res)