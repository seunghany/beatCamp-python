from dataclasses import dataclass

@dataclass
class Entity:
    
    context: str = '/Users/seung/SbaProjects/beatCamp-python'
    fname: str = ''
    train: object = None
    text: object = None
    id: str = ''
    label: str = ''