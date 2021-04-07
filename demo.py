import ink

# 分词
print("====== 分词 =======")
sents = [  "我爱北京天安门" ]
print("bert:", ink.cws.get_cws("bert")(sents))
print("thulac:", ink.cws.get_cws("thulac")(sents))
print("\n\n")

print("====== NER =======")
sents = [  "我爱北京天安门" ]
print("bert:", ink.ner.get_ner("bert")(sents))

print("====== POS Tagging =======")
sents = [  "我爱北京天安门" ]
print("bert:", ink.postagging.get_pos_tagging("bert")(sents))

print("====== POS Tagging =======")
sents = [("3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", (30, 33)),
    ("张淑芳老人记得照片是在工人文化宫照的，而且还是在一次跳完集体舞后拍摄的。但摄影师是谁，照片背后的字是谁写的，已经找寻不到答案了。", (22, 24))]
print("bert:", ink.typing.get_typing("bert")(sents))
