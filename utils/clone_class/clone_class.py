from typing import List

class CloneClass:
    """
    表示一个代码克隆类
    """
    def __init__(self, representative_function_id: int):
        # 作为一个类的代表函数 ID
        self.representative_function_id: int = representative_function_id
        
        # 疑似为这个克隆类的函数 ID 列表
        self.suspicious_functions: List[int] = []
        
        # 已确定为这个克隆类的函数 ID 列表
        self.confirmed_functions: List[int] = []
