import oknlp

# 分词
print("====== 分词 =======")
sents = ["我爱北京天安门"]

model = oknlp.cws.get_by_name("bert")
print("bert:", model(sents))
model = oknlp.cws.get_by_name("thulac")
print("thulac:", model(sents))
print("\n\n")

print("====== NER =======")
sents = ["我爱北京天安门"]
model = oknlp.ner.get_by_name("bert")
print("bert:", model(sents))

print("====== POS Tagging =======")
sents = ["我爱北京天安门"]
model = oknlp.postagging.get_by_name("bert")
print("bert:", model(sents))

print("====== Typing =======")
sents = [("3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", (30, 33)),
    ("张淑芳老人记得照片是在工人文化宫照的，而且还是在一次跳完集体舞后拍摄的。但摄影师是谁，照片背后的字是谁写的，已经找寻不到答案了。", (22, 24))]
model = oknlp.typing.get_by_name("bert")
print("bert:", model(sents))
