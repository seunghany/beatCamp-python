# 기본 entity 파일.
from dataclasses import dataclass
# Path 는 바뀜
"""
context = "/Users/seung/SbaProjects/beatCamp-python"
fname = "/titanic/data"
"""


@dataclass
class FileReader:
    context: str = ''
    fname: str = ''
    train: object = None
    test: object = None
    id: str = ''
    label: str = ''