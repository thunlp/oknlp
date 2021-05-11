import string

ch_punc = r"！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
en_punc = string.punctuation
punc = set(ch_punc + en_punc)


def split_text_by_punc(text, max_length):
    """（用标点符号）分割一个字符串，分割后的每个字符串长度均不大于max_length，返回分割后的字符串列表
    """
    range_list = []
    begin, end = 0, 0
    while end <= len(text):
        if end - begin == max_length or text[end] in punc or end == len(text):
            range_list.append((begin, end))
            begin = end
        end += 1
    start = 0
    split_text_list = []
    for (begin, end) in range_list:
        if end - start > max_length:
            split_text_list.append(text[start: begin])
            start = begin
    split_text_list.append(text[start:])


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
