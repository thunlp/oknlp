import string


def split_text_by_punc(text, max_length):
    """分割一个字符串，分割后的每个字符串长度均不大于max_length，返回分割后的字符串列表
    """
    if len(text) <= max_length:
        return [text]
    ch_punc = r"！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    en_punc = string.punctuation
    punc = set(ch_punc + en_punc)
    for i in range(len(text) - 1, -1, -1):
        if text[i] in punc:
            return split_text_by_punc(text[:i], max_length) + [text[i]] + split_text_by_punc(text[i+1:], max_length)
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]


def split_text_list(text_list, max_length):
    """将字符串列表中的每个字符串进行分割，分割后的每个字符串长度均不大于max_length，返回分割后的字符串列表、表示是否结束的列表
    """
    split_text_list = []
    is_end = []
    for text in text_list:
        split_text_list += split_text_by_punc(text, max_length)
        is_end += [False] * (len(split_text_list) - len(is_end) - 1)
        is_end.append(True)
    return split_text_list, is_end


def merge_result(result_list, is_end_list):
    ans = []
    tmp = []
    for result, is_end in zip(result_list, is_end_list):
        tmp += result
        if is_end:
            ans.append(tmp)
            tmp = []
    return ans
