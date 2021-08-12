from typing import List
from ..abc import Algorithm
from ...utils.process_io import split_text_list, merge_result

class BaseCWS(Algorithm):

    def __call__(self, sents: List[str],max_len = 12) -> List[List[str]]:
        """
        Args:
            sents: 输入的句子列表。
        Returns:
            返回一个和输入相同长度的列表，列表中的每一项都是对应输入的分词结果。
        
        更多信息请参考 :doc:`中文分词 - 示例</examples/cws>`

        Examples:    
            >>> import oknlp
            >>> cws = oknlp.cws.get_by_name()
            >>> cws(['我爱北京天安门'])
            [['我', '爱', '北京', '天安门']]
        
        """
       
        return super().__call__(sents)
        
        
