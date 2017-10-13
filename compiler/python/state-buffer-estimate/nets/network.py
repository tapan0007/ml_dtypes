from abc             import ABCMeta, abstractmethod

from utils.consts    import  *
from utils.funcs     import kstr
import layers.layer
from schedule.scheduler      import Scheduler

##########################################################
class Network(object):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self):
        self.__Layers = [ ]
        self.__LayerNumMajor = -1
        self.__LayerNumMinor = 0
        self.__Levels = None
        self.__CurrLayerId = 0

    #-----------------------------------------------------------------
    def rLevels(self, levels):
        self.__Levels = levels

    #-----------------------------------------------------------------
    def gLayers(self):
        return self.__Layers

    #-----------------------------------------------------------------
    def gLayer(self, idx):
        return self.__Layers[idx]

    #-----------------------------------------------------------------
    def addLayer(self, layer):
        assert(layer)
        assert( isinstance(layer, layers.layer.Layer) )
        layer.rLayerId(self.__CurrLayerId); self.__CurrLayerId += 1
        self.__Layers.append(layer)
        t = layer.gLayerType()
        if (t == LAYER_TYPE_DATA or t == LAYER_TYPE_CONV or t == LAYER_TYPE_FULL):
            self.__LayerNumMajor += 1
            self.__LayerNumMinor = 0
            numStr = str(self.__LayerNumMajor)
        else:
            numStr = str(self.__LayerNumMajor) + "." + str(self.__LayerNumMinor)
            self.__LayerNumMinor += 1
        layer.rNumberStr(numStr)

    #-----------------------------------------------------------------
    def gNumberLayers(self):
        return len(self.__Layers)

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gLayer(0).gLayerType() == LAYER_TYPE_DATA)
        numLayers = self.gNumberLayers()

        for layer in self.gLayers(): # self.__Layers:
            layer.verify()



    #-----------------------------------------------------------------
    def printDot(self):
        f1=open(self.gName()+".dot", 'w')

        graphName = self.gName().replace("-", "_").replace(".", "_")
        print >>f1, 'digraph', graphName, "{"

        for layer in self.gLayers():
            print >>f1, '  ', layer.gDotIdLabel()

        print >>f1

        for layer in self.__Layers:
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
        self.m_PrevLayer = None

        for layer in self.__Layers:
            if layer.gDenseBlockStart() >= 0:
                if not prevNl:
                    print
                print ">>> Starting dense block " + str(layer.gDenseBlockStart())
            elif layer.gTranBlockStart() >= 0:
                if not prevNl:
                    print
                print ">>> Starting tran block " + str(layer.gTranBlockStart())

            inStateSize = layer.gRawInputStateSize()
            outStateSize = layer.gRawOutputStateSize()
            totalStateSize = inStateSize + outStateSize
            if totalStateSize > maxStateSize:
                maxStateSize = totalStateSize

            numStr = layer.gNumberStr()
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

            self.m_PrevLayer =layer

        print "Max state size =", kstr(maxStateSize)


    #-----------------------------------------------------------------
    def printLevels(self):
        for level in self.__Levels:
            for layer in level.gLayers():
                print layer.gNameWithSched(),
            print


    #-----------------------------------------------------------------
    @abstractmethod
    def construct(self):
        assert(False)

    #-----------------------------------------------------------------
    @abstractmethod
    def gName(self):
        assert(False)

    #-----------------------------------------------------------------
    def gSchedLayers(self):
        #-------------------------------------------------------------
        class SchedLayerForwIter(object):
            def __init__(self, startLayer):
                self.__CurrLayer = startLayer

            def __iter__(self):
                return self

            def next(self):
                currLayer = self.__CurrLayer
                if not currLayer:
                    raise StopIteration()

                nextLayer = currLayer.gNextSchedLayer()
                self.__CurrLayer = nextLayer
                return currLayer



        return SchedLayerForwIter(self.__Layers[0])


    #-----------------------------------------------------------------
    def printSched(self):
        for layer in self.gSchedLayers():
            sbPreds = ""
            first=True
            for sbLayer in layer.gPrevSbLayers():
                s = sbLayer.gNameNum()
                if not first:
                    s = "," + s
                first=False
                sbPreds += s

            sb = "SB" if layer.qStoreInSB() else "sb"
            print layer.gNameWithSchedMem() + " " + sb + "=[" + sbPreds  + "]"

