from abc             import ABCMeta, abstractmethod

from utils.consts    import  *
from utils.funcs     import kstr
import layers.layer

##########################################################
class Network(object):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self):
        self.m_Layers =[ ]

    def gLayer(self, idx):
        return self.m_Layers[idx]

    #-----------------------------------------------------------------
    def addLayer(self, layer):
        assert(layer)
        assert( isinstance(layer, layers.layer.Layer) )
        self.m_Layers.append(layer)
        layer.rIndex(len(self.m_Layers) - 1)

    #-----------------------------------------------------------------
    def gNumberLayers(self):
        return len(self.m_Layers)

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.m_Layers[0].gLayerType() == LAYER_TYPE_DATA)
        numLayers = self.gNumberLayers()

        for layer in self.m_Layers:
            layer.verify()


##        # Check connections between two consecutive layers.
##        # Special case for Concat layers
##        for layerIdx in range(numLayers-1):
##            layer1 = self.m_Layers[layerIdx]
##            layer2 = self.m_Layers[layerIdx+1]
##            if layer2.gLayerType() != LAYER_TYPE_CONCAT:
##                assert(layer1.gOfmapDesc() == layer2.gIfmapDesc())
##            else:
##                earlier_layer = layer2.gEarlierLayer()
##                mapSize = layer1.gOfmapSize()
##                assert(earlier_layer.gOfmapSize() == mapSize
##                          and layer2.gIfmapSize() == mapSize)
##                assert (layer1.gNumOfmaps() + earlier_layer.gNumOfmaps()
##                    ==  layer2.gNumIfmaps())


    def printDot(self):
        f1=open(self.gName()+".dot", 'w')

        graphName = self.gName().replace("-", "_").replace(".", "_")
        print >>f1, 'graph', graphName, "{"

        for layer in self.m_Layers:
            print >>f1, '  ', layer.gDotIdLabel()

        print >>f1

        for layer in self.m_Layers:
            for nextLayer in layer.gNextLayers():
                print >>f1, '  ', layer.gDotId(), '->', nextLayer.gDotId(), ';'

        print >>f1, '}'
        print >>f1

    #-----------------------------------------------------------------
    def printMe(self):
        prevNl = False
        maxStateSize = 0
        layerNumMajor = 0
        layerNumMinor = 0

        for layer in self.m_Layers:
            if layer.gDenseBlockStart() >= 0:
                if not prevNl:
                    print
                print ">>> Starting dense block " + str(layer.gDenseBlockStart())
            elif layer.gTranBlockStart() >= 0:
                if not prevNl:
                    print
                print ">>> Starting tran block " + str(layer.gTranBlockStart())

            totalStateSize = layer.gSingleBatchTotalStateSize()
            if totalStateSize > maxStateSize:
                maxStateSize = totalStateSize

            if layer.gLayerType() == LAYER_TYPE_DATA or layer.gLayerType() == LAYER_TYPE_CONV:
                numStr = str(layerNumMajor); layerNumMajor += 1; layerNumMinor = 0
            else:
                numStr = str(layerNumMajor) + "." + str(layerNumMinor); layerNumMinor += 1

            print (numStr + " " + str(layer))
            layer.m_NumStr = numStr

            prevNl = False
            if layer.gDenseBlockEnd() >= 0:
                print "<<< Ending dense block " + str(layer.gDenseBlockEnd())
                print
                prevNl = True
            elif layer.gTranBlockEnd() >= 0:
                print "<<< Ending tran block " + str(layer.gTranBlockEnd())
                print
                prevNl = True

        print "Max state size =", kstr(maxStateSize)

    @abstractmethod
    def construct(self):
        assert(False)

    @abstractmethod
    def gName(self):
        assert(False)


