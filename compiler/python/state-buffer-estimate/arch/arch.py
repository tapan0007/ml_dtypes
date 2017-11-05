from .psumbuffer         import PsumBuffer
from .pearray            import PeArray
from .statebuffer        import StateBuffer
from .poolingeng         import PoolingEng
from .activationeng      import ActivationEng

##########################################################
class Arch(object):

    #-----------------------------------------------------------------
    def __init__(self):

        numberPeRows            = 128
        numberPeColumns         = 64
        numberPsumBanks         = 4
        numberPsumBankEntries   = 256
        sbPartitionsSize        = 12 * 1024 * 1024 // numberPeRows  ## 12 MB
        sbPartitionsSize        =  8 * 1024 * 1024 // numberPeRows  ##  8 MB

        self.__PeArray          = PeArray(numberPeRows, numberPeColumns)
        self.__PsumBuffer       = PsumBuffer(self.gPeArray(), numberPsumBanks,
                                             numberPsumBankEntries)
        self.__PoolingEng       = PoolingEng(self.gPsumBuffer())
        self.__ActivationEng    = ActivationEng(self.gPsumBuffer())
        self.__StateBuffer      = StateBuffer(self.gPeArray(), sbPartitionsSize)

    #-----------------------------------------------------------------
    def gPeArray(self):
        return self.__PeArray

    #-----------------------------------------------------------------
    def gStateBuffer(self):
        return self.__StateBuffer

    #-----------------------------------------------------------------
    def gPsumBuffer(self):
        return self.__PsumBuffer

    #-----------------------------------------------------------------
    def gPoolingEng(self):
        return self.__PoolingEng

    #-----------------------------------------------------------------
    def gActivationEng(self):
        return self.__ActivationEng


    #-----------------------------------------------------------------
    def gNumberPeArrayRows(self):
        return self.__PeArray.gNumberRows()

    #-----------------------------------------------------------------
    def gNumberPeArrayColumns(self):
        return self.__PeArray.gNumberColumns()



    #-----------------------------------------------------------------
    def gNumberPsumBanks(self):
        return self.gPsumBuffer().gNumberBanks()

    #-----------------------------------------------------------------
    def gPsumBankEntries(self):
        return self.gPsumBuffer().gNumberBankEntries()



