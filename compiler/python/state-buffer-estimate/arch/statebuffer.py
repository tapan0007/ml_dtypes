from pearray import PeArray

##########################################################
class StateBuffer(object):
    #-----------------------------------------------------------------
    def __init__(self, peArray, partitionSizeInBytes):
        assert(isinstance(peArray, PeArray))

        self.__NumberPartitions = peArray.gNumberRows()
        assert(self.__NumberPartitions > 0)
        assert(self.__NumberPartitions & (self.__NumberPartitions - 1) == 0) ## power of 2
        self.__PartitionSizeInBytes = partitionSizeInBytes
        self.__TotalSizeInBytes     = self.__NumberPartitions * self.__PartitionSizeInBytes

    #-----------------------------------------------------------------
    def gNumberPartitions(self):
        return self.__NumberPartitions

    #-----------------------------------------------------------------
    def gPartitionSizeInBytes(self):
        return self.__PartitionSizeInBytes

    #-----------------------------------------------------------------
    def gTotalSizeInBytes(self):
        return self.__TotalSizeInBytes

    #-----------------------------------------------------------------
    def qLittleEndian(self):
        return True

    #-----------------------------------------------------------------
    def gFirstAddressInBytes(self):
        return 0

    #-----------------------------------------------------------------
    def gPartitionStartAddressInBytes(self, partNum):
        assert(0 <= partNum and partNum < self.gNumberPartitions())
        return self.gFirstAddress() + partNum * self.gPartitionSizeInBytes()

    
