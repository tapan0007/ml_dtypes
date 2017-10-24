from arch.arch import Arch

class StateBufferMgr(object):
    def __init__(self, arch):
        assert(isinstance(arch, Arch))
        self.__Arch = arch

        sbuf = self.__StateBuffer = arch.gStateBuffer()
        self.__PartitionSize = sbuf.gPartitionSizeInBytes()
        self.__FirstSbAddress = sbuf.gFirstAddressInBytes()

        self.__FirstFreeStart = self.__FirstSbAddress


    #def gMem(self, amountInBytes):
        
