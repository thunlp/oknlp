# coding=UTF-8
import thulac
from ink.data import load
def lac_cws(sents):
    model_path = load('models')
    print(model_path)
    result = []
    for sent in sents:
        tmp = thulac.thulac(sent,model_path)
        result.append((tmp.decode('utf-8')))
    return result
if __name__ =='__main__':
    res = lac_cws(['我爱北京天安门'])
    print(res)