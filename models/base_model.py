from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, opt):
    	self.opt = opt