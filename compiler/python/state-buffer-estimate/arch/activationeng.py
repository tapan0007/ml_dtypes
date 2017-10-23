from psumbuffer import PsumBuffer

class ActivationEng(object):
    def __init__(self, psumBuffer):
        assert(isinstance(psumBuffer, PsumBuffer))
        self.__Width = psumBuffer.gNumberColumns()

    def gWidth(self):
        return self.__Width
        
