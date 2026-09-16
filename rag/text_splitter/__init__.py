from .chinese_recursive_splitter import *

# 条件导入需要额外依赖的模块
try:
    from .semantic_splitter import *
except ImportError:
    pass