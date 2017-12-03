#from abc             import ABCMeta, abstractmethod

from utils.consts    import  *
##from utils.funcs     import kstr
from utils.datatype  import *
from utils.debug  import breakFunc

import layers.layer
from layers.layer           import Layer
from layers.convlayer       import ConvLayer
from layers.maxpoollayer    import MaxPoolLayer
from layers.avgpoollayer    import AvgPoolLayer
from layers.datalayer       import DataLayer
from layers.addlayer        import AddLayer
from layers.fulllayer       import FullLayer
from layers.softmaxlayer    import SoftMaxLayer
from layers.batchnormlayer  import BatchNormLayer
from layers.relulayer       import ReluLayer
from layers.tanhlayer       import TanhLayer
from layers.concatlayer     import ConcatLayer

from schedule.scheduler      import Scheduler

##########################################################
##class Network(object, metaclass = abc.ABCMeta):
class Network(object):
    net_name_key  = "net_name"
    data_type_key = "data_type"
    layers_key    = "layers"

    #-----------------------------------------------------------------
    class SchedLayerForwRevIter(object):
        def __init__(self, startLayer, forw):
            self.__Forw = forw
            self.__CurrLayer = startLayer

        def __iter__(self):
            return self

        def __next__(self):
            currLayer = self.__CurrLayer
            if not currLayer:
                raise StopIteration()

            if self.__Forw:
                nextLayer = currLayer.gNextSchedLayer()
            else:
                nextLayer = currLayer.gPrevSchedLayer()

            self.__CurrLayer = nextLayer
            return currLayer

    #-----------------------------------------------------------------
    def __init__(self, dataType, netName):
        assert(isinstance(dataType, DataType))
        self.__Name = netName
        self.__Layers = [ ]
        self.__LayersByName = {}
        self.__LayerNumMajor = -1
        self.__LayerNumMinor = 0
        self.__Levels = None
        self.__CurrLayerId = 0
        self.__DoBatching = False
        self.__DataType = dataType
        self.__UseDimList = True
        self.__Constructed = False

    #-----------------------------------------------------------------
    def qConstructed(self):
        return self.__Constructed

    #-----------------------------------------------------------------
    def gUseDimList(self):
        return self.__UseDimList

    #-----------------------------------------------------------------
    def gDataType(self):
        return self.__DataType

    #-----------------------------------------------------------------
    def qDoBatching(self):
        return self.__DoBatching

    #-----------------------------------------------------------------
    def rDoBatching(self, batch):
        self.__DoBatching = batch

    #-----------------------------------------------------------------
    def gLevels(self):
        return self.__Levels

    def rLevels(self, levels):
        self.__Levels = levels

    #-----------------------------------------------------------------
    def gLayers(self):
        return self.__Layers

    #-----------------------------------------------------------------
    def gLayer(self, idx):
        return self.__Layers[idx]

    #-----------------------------------------------------------------
    def gLayerByName(self, name):
        return self.__LayersByName[name]

    #-----------------------------------------------------------------
    def addLayer(self, layer):
        assert(layer)
        assert( isinstance(layer, layers.layer.Layer) )
        layer.rLayerId(self.__CurrLayerId); self.__CurrLayerId += 1
        self.__Layers.append(layer)
        self.__LayersByName[layer.gName()] = layer
        if layer.qDataLayer() or layer.qConvLayer() or layer.qFullLayer():
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
        assert(self.gLayer(0).qDataLayer())
        numLayers = self.gNumberLayers()

        for layer in self.gLayers(): # self.__Layers:
            layer.verify()





    #-----------------------------------------------------------------
    def gName(self):
        return self.__Name

    #-----------------------------------------------------------------
    def gSchedLayers(self):
        #-------------------------------------------------------------
        return Network.SchedLayerForwRevIter(self.__Layers[0], True)

    #-----------------------------------------------------------------
    def gReverseSchedLayers(self):
        return Network.SchedLayerForwRevIter(self.__Layers[-1], False)

    #-----------------------------------------------------------------
    def gJson(self):
        jsonDict = { 
            Network.net_name_key : self.gName(),
            Network.data_type_key : self.__DataType.gName()
        }
        json_layers = []
        for layer in self.gLayers():
            layer_json = layer.gJson()
            assert layer_json 
            json_layers.append(layer_json)
        jsonDict[Network.layers_key] = json_layers
        return jsonDict

    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(cls, jsonDict):
        netName = jsonDict[Network.net_name_key]
        dt = jsonDict[Network.data_type_key]
        if dt == "int8":
            dataType = DataTypeInt8()
        elif dt == "int16":
            dataType = DataTypeInt16()
        elif dt == "float16":
            dataType = DataTypeFloat16()
        nn = Network(dataType, netName)

        name2class = {}
        LeafClasses = [
            ConvLayer, MaxPoolLayer, DataLayer, 
            AddLayer, AvgPoolLayer, FullLayer,
            SoftMaxLayer, BatchNormLayer,
            ReluLayer, TanhLayer,
            ConcatLayer,
        ]
        for leafClass in LeafClasses:
            name2class[leafClass.gTypeStr()] = leafClass

        layerDicts = jsonDict[Network.layers_key]

        for layerDict in layerDicts:
            layerType = layerDict[Layer.type_key]
            if layerType == "Full":
                breakFunc(3)
            layer = None
            cls = name2class[layerType]
            if cls:
                layer = cls.constructFromJson(layerDict, nn)
            else:
                print(layerType)
                assert False

        nn.__Constructed = True
        return nn


