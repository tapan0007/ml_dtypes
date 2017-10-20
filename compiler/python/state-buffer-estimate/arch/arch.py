from abc             import ABCMeta, abstractmethod

#from arch.psumbuffer import PsumBuffer
from psumbuffer import PsumBuffer
from pearray import PeArray
from statebuffer import StateBuffer

##########################################################
class Arch(object):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self):
        self.__PeArray    = PeArray()
        self.__PsumBuffer = PsumBuffer()


    #-----------------------------------------------------------------
    def gNumberPeArrayRows(self):
        return self.__PeArray.gNumberRows()

    #-----------------------------------------------------------------
    def gNumberPeArrayColumns(self):
        return self.__PeArray.gNumberColumns()



    #-----------------------------------------------------------------
    def gNumberPsumBanks(self):
        return self.__PsumBuffer.gNumberBanks()

    #-----------------------------------------------------------------
    def gPsumBankEntries(self):
        return self.__PsumBuffer.gNumberBankEntries()





