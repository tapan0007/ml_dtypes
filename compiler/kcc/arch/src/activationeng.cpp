"""
The module describing Activation Engine class
"""
from .psumbuffer import PsumBuffer

class ActivationEng(object):
    """
    Activation Engine
    Important parameters are
        width :
        memory mapped addresses
    """
    def __init__(self, psumBuffer):
        assert isinstance(psumBuffer, PsumBuffer)
        self.__Width = psumBuffer.gNumberColumns()

    def gWidth(self):
        return self.__Width

    def gInstructionRamStartInBytes(self):
        return 0x001F00000

    def gInstructionRamEndInBytes(self):
        return 0x001F03FFF
