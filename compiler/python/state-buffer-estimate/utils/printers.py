import json

from utils.funcs     import kstr
from utils.consts    import *
from utils.datatype  import *
from layers.layer import Layer
from nets.network import Network

class Printer(object):

    #-----------------------------------------------------------------
    def __init__(self, netwk):
        self.__Network = netwk

    #-----------------------------------------------------------------
    def printNetwork(self):
        ntwk = self.__Network
        prevNl = False
        maxStateSize = 0
        layerNumMajor = 0
        layerNumMinor = 0
        self.__PrevLayer = None

        for layer in ntwk.gLayers():
            if layer.gDenseBlockStart() >= 0:
                if not prevNl:
                    print
                print (">>> Starting dense block " + str(layer.gDenseBlockStart()))
            elif layer.gTranBlockStart() >= 0:
                if not prevNl:
                    print
                print(">>> Starting tran block " + str(layer.gTranBlockStart()))

            if layer.qStoreInSB():
                inStateSize = layer.gInputStateMemWithoutBatching()
                outStateSize = layer.gOutputStateMemWithoutBatching()
                totalStateSize = inStateSize + outStateSize
                if totalStateSize > maxStateSize:
                    maxStateSize = totalStateSize
            else:
                inStateSize = layer.gInputSize()
                outStateSize = layer.gOutputSize()

            numStr = layer.gNumberStr()
            print (numStr + " " + str(layer))
            layer.m_NumStr = numStr

            prevNl = False
            if layer.gDenseBlockEnd() >= 0:
                print("<<< Ending dense block " + str(layer.gDenseBlockEnd()))
                print
                prevNl = True
            elif layer.gTranBlockEnd() >= 0:
                print("<<< Ending tran block " + str(layer.gTranBlockEnd()))
                print
                prevNl = True

            self.__PrevLayer =layer

        print("Max state size =", kstr(maxStateSize))

    #-----------------------------------------------------------------
    def printDot(self):
        ntwk = self.__Network
        f1=open(netwk.gName()+".dot", 'w')

        graphName = netwk.gName().replace("-", "_").replace(".", "_")
        print >>f1, 'digraph', graphName, "{"

        for layer in netwk.gLayers():
            print >>f1, '  ', layer.gDotIdLabel()

        print >>f1

        for layer in netwk.__Layers:
            for nextLayer in layer.gNextLayers():
                print >>f1, '  ', layer.gDotId(), '->', nextLayer.gDotId(), ';'

        print >>f1, '}'
        print >>f1


    #-----------------------------------------------------------------
    def printLevels(self):
        ntwk = self.__Network
        for level in ntwk.gLevels():
            for layer in level.gLayers():
                print(layer.gNameWithSched(),)
            print

    #-----------------------------------------------------------------
    def printSched(self):
        ntwk = self.__Network
        dataType = ntwk.gDataType()
        print(ntwk.gName(), ": data type=", dataType.gName(), " data type size=", dataType.gSizeInBytes())
        memHeader = (SCHED_MEM_FORMAT) % (
            "Layer", "Ofmap", "In", "Out",
            "Residue",
            "Batch",
            "BatchDlt",
            )
        lineFmt = ("%-70s  %s")
        fullHeader = (lineFmt) % (memHeader, "SB predecessors")
        print(fullHeader)
        hasRelu = False
        lastWasAdd = False

        for layer in ntwk.gSchedLayers():
            if layer.qReluLayer():
                hasRelu = True
            sbPreds = ""
            first=True
            for sbLayer in layer.gPrevSbLayers():
                s = sbLayer.gName()
                if not first:
                    s = "," + s
                first=False
                sbPreds += s

            if sbPreds == "":
                sbPreds = "()"
            sb = "SB" if layer.qStoreInSB() else "--"
            ss = (lineFmt) % (layer.gNameWithSchedMem(), "[" + sb + "]=" + sbPreds)

            ifaddr = layer.gIfmapAddress()
            ofaddr = layer.gOfmapAddress()
            waddr  = layer.gWeightAddress()

            if ifaddr != None or ofaddr != None or waddr != None:
                ss += " {"
                b = False
                if ifaddr != None:
                    ss += "i=" + str(ifaddr)
                    b = True
                if ofaddr != None:
                    if b: ss += ", "
                    ss += "o=" + str(ofaddr)
                    b = True
                if waddr != None:
                    if b: ss += ", "
                    ss += "w=" + str(waddr)
                    b = True
                ss += "}"

            print(ss)
            if hasRelu:
                if lastWasAdd and layer.qReluLayer():
                    print
            else:
                if layer.qAddLayer() or layer.qPoolLayer():
                    print

            lastWasAdd = layer.qAddLayer()

        print(fullHeader)

    def printJson(self, obj, filename):
        obj_json = json.dumps(obj.gJson(), sort_keys=False, indent=4, separators=(',', ': '))
        with open(filename, "w") as f:
            f.write(obj_json)

