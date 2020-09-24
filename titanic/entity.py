from dataclasses import dataclass


@dataclass
class Entity:
    context: str = "/Users/seung/SbaProjects/beatCamp-python/"
    fname: str = ''
    train: object = None
    test: object = None
    id: str = ''
    label: str = ''

    # def __int__(self, context, fname, train, test, id, label):
    #     self._context = context   # _ 1개는 defualt의 의미, __2개는 private 접근 의미
    #     self._fname = fname
    #     self._train = train
    #     self._test = test
    #     self._id = id
    #     self._label = label

    # @property
    # def context(self) -> str: # return type 이 str 이란 의미
    #     return self._context

    # @context
    # def context(self, context):
    #     self._context = context

    # @property
    # def name(self):
    #     return self._fname

    # @context
    # def name(self, fname):
    #     self._fname = fname
    
    # @property
    # def id(self):
    #     return self._id

    # @context
    # def id(self, id):
    #     self._id = id

    # @property
    # def label(self):
    #     return self._label

    # @context
    # def label(self, label)
    #     self._label = label
