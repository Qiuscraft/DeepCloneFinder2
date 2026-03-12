from typing import List
from utils.java_code.function_info import FunctionInfo

class CloneClass:
    """
    表示一个代码克隆类
    """
    def __init__(self, representative_function: FunctionInfo):
        # 作为一个类的代表函数
        self.representative_function: FunctionInfo = representative_function
        
        # 疑似为这个克隆类的函数列表
        self.suspicious_functions: List[FunctionInfo] = []
        
        # 已确定为这个克隆类的函数列表
        self.confirmed_functions: List[FunctionInfo] = []
