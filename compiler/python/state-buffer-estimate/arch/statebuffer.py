
##########################################################
class StateBuffer(object):
    #-----------------------------------------------------------------
    def __init__(self, numPartitions):
        totalSize = 12 * 1024 * 1024  ## 12 MBs in bytes
        assert(numPartitions > 0)
        assert(numPartitions & (numPartitions - 1) == 0) ## 2^M
        assert(totalSize % numPartitions == 0)

        self.__TotalSizeInBytes     = totalSize
        self.__NumberPartitions     = numPartitions
        self.__PartitionSizeInBytes = totalSize / numPartitions

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
    def gPartitionFirstAddressInBytes(self, partNum):
        assert(0 <= partNum and partNum < self.gNumberPartitions())
        return self.gFirstAddress() + partNum * self.gPartitionSizeInBytes()

    
