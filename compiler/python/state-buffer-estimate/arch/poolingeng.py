from .psumbuffer import PsumBuffer

class PoolingEng(object):
    def __init__(self, psumBuffer):
        assert(isinstance(psumBuffer, PsumBuffer))
        self.__Width = psumBuffer.gNumberColumns()

    def gWidth(self):
        return self.__Width

    def gInstructionRamStartInBytes(self):
        return 0x001E00000

    def gInstructionRamEndInBytes(self):
        return 0x001E03FFF

