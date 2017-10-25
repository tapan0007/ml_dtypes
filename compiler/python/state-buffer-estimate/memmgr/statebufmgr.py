import nets.network 
from arch.arch import Arch

class StateBufferMgr(object):
    def __init__(self, arch, ntwk):
        assert(isinstance(arch, Arch))
        assert(isinstance(ntwk, nets.network.Network))
        self.__Arch = arch
        self.__Network = ntwk

        sbuf = self.__StateBuffer = arch.gStateBuffer()
        self.__PartitionSize = sbuf.gPartitionSizeInBytes()
        self.__FirstSbAddress = sbuf.gFirstAddressInBytes()

        self.__FirstFreeStart = self.__FirstSbAddress
        self.calcLayerFmapAddresses()


    def calcLayerFmapAddresses(self):
        maxNumWeights = 0
        for layer in self.__Network.gLayers():
            if layer.gNumberWeights() > maxWeightSize:
                maxNumWeights = layer.gNumberWeights() 

        self.__MaxNumberWeights = maxNumWeights


        ## first layer is Data layer and will have no ifmap
        ofmapAddress = ifmapAddress = None

        for layer in self.__Network.gLayers():
            if layer.qDataLayer():
                assert(ifmapAddress == None and ofmapAddress == None)
                ofmapAddress = self.__FirstSbAddress + self.__MaxNumberWeights
                layer.rIfmapAddress(ifmapAddress)
                layer.rOfmapAddress(ofmapAddress)
            elif layer.qConvLayer():
                assert(ofmapAddress != None)
                prevIfmapAddress = ifmapAddress
                prevOfmapAddress = ofmapAddress
                ifmapAddress = ofmapAddress

                if prevIfmapAddress == None: ## after data layer
                    ##         Weights | prevOfmap | ... | ...      
                    ofmapAddress = self.__FirstSbAddress + self.__PartitionSize - layer.gOutputStateMemWithBatching()
                else:
                    if prevIfmapAddress < prevOfmapAddress:
                        ##     Weights | prevIfmap | ... | prevOfmap
                        ##             | Ofmap   
                        ofmapAddress = self.__FirstSbAddress + self.__MaxNumberWeights
                    else:
                        ##     Weights | prevOfmap | ... | prevIfmap
                        ##                             | Ofmap   
                        ofmapAddress = self.__FirstSbAddress + self.__PartitionSize - layer.gOutputStateMemWithBatching()

                layer.rIfmapAddress(ifmapAddress)
                layer.rOfmapAddress(ofmapAddress)


