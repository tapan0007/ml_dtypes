from .pearray import PeArray

##########################################################
class PsumBuffer(object):

    #-----------------------------------------------------------------
    def __init__(self, peArray, numberBanks, numberBankEntries):
        assert isinstance(peArray, PeArray)
        self.__NumberColumns        = peArray.gNumberColumns()
        self.__NumberBanks          = numberBanks
        self.__NumberBankEntries    = numberBankEntries
        self.__BankEntrySizeInBytes = 64 ## ???

    #-----------------------------------------------------------------
    def gNumberBanks(self):
        return self.__NumberBanks

    #-----------------------------------------------------------------
    def gNumberBankEntries(self):
        return self.__NumberBankEntries

    #-----------------------------------------------------------------
    def gBankEntrySizeInBytes(self):
        return self.__BankEntrySizeInBytes

    #-----------------------------------------------------------------
    def gNumberColumns(self):
        return self.__NumberColumns

    def gAddress(self):
        return 0x001800000
