

##########################################################
class PsumBuffer(object):

    #-----------------------------------------------------------------
    def __init__(self):
        self.__NumberBanks          = 4
        self.__NumberBankEntries    = 256
        self.__BankEntrySizeInBytes = 64 ## ???

    #-----------------------------------------------------------------
    def gNumberBanks(self):
        return self.__NumberBanks

    #-----------------------------------------------------------------
    def gNumberBankEntries(self):
        return self.__NumberBankEntries

    def gBankEntrySizeInBytes(self):
        return self.__BankEntrySizeInBytes

