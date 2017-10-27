from utils.debug    import breakFunc
import nets.network 
from arch.arch import Arch

##########################################################
class StateBufferMgr(object):
    #-----------------------------------------------------------------
    def __init__(self, arch, ntwk):
        assert(isinstance(arch, Arch))
        assert(isinstance(ntwk, nets.network.Network))
        self.__Arch = arch
        self.__Network = ntwk

        sbuf = self.__StateBuffer = arch.gStateBuffer()
        self.__PartitionSize = sbuf.gPartitionSizeInBytes()
        self.__FirstSbAddress = sbuf.gFirstAddressInBytes()

        self.__FirstFreeStart = self.__FirstSbAddress


    #-----------------------------------------------------------------
    def calcOneLayerFmapAddresses(self, layer):
        if not layer.qStoreInSB():
            return

        prevOfmapAddress = self.__OfmapAddress 
        prevIfmapAddress = self.__IfmapAddress 

        if layer.qDataLayer():
            assert(prevIfmapAddress == None and prevOfmapAddress == None)
            ifmapAddress = None
            ofmapAddress = self.__FirstSbAddress + self.__MaxNumberWeightsPerPart
        else:
            assert(prevOfmapAddress != None)
            ifmapAddress = prevOfmapAddress

            if prevIfmapAddress == None: ## after data layer
                ##         Weights | prevOfmap | ... | ...      
                ## need to get batching memory per partition
                ofmapAddress = self.__FirstSbAddress + self.__PartitionSize - layer.gOutputStateMemWithBatching()
            elif prevIfmapAddress < prevOfmapAddress:
                ##     Weights | prevIfmap | ... | prevOfmap
                ##             | Ofmap   
                ofmapAddress = self.__FirstSbAddress + self.__MaxNumberWeightsPerPart
            else:
                ##     Weights | prevOfmap | ... | prevIfmap
                ##                             | Ofmap   
                ofmapAddress = self.__FirstSbAddress + self.__PartitionSize - layer.gOutputStateMemWithBatching()

        layer.rIfmapAddress(ifmapAddress)
        layer.rOfmapAddress(ofmapAddress)
        layer.rWeightAddress(self.__FirstSbAddress)

        self.__OfmapAddress = ofmapAddress 
        self.__IfmapAddress = ifmapAddress 

    #-----------------------------------------------------------------
    def calcLayerFmapAddresses(self):
        maxNumWeightsPerPart = 0
        for layer in self.__Network.gLayers():
            numWeights = layer.gNumberWeightsPerPartition()
            if numWeights > maxNumWeightsPerPart :
                maxNumWeightsPerPart = numWeights

        breakFunc(0)
        self.__MaxNumberWeightsPerPart = maxNumWeightsPerPart


        ## first layer is Data layer and will have no ifmap
        self.__OfmapAddress = self.__IfmapAddress = None

        for layer in self.__Network.gLayers():
            self.calcOneLayerFmapAddresses(layer)


