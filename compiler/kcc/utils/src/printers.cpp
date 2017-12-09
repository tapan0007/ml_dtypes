//import json

//from utils.funcs     import kstr
#include "consts.hpp"
#include "datatype.hpp"
#include "layer.hpp"
#include "network.hpp"

//--------------------------------------------------------
void
Printer::printNetwork()
{
    Network* ntwk = m_Network;
    bool prevNl = False;
    StateBufferAddress maxStateSize = 0;
    int layerNumMajor = 0;
    int layerNumMinor = 0;
    m_PrevLayer = nullptr;

    for (auto layer : ntwk.gLayers()) {
        if (layer->gDenseBlockStart() >= 0) {
            if (!prevNl) {
                cout << "\n";
            }
            cout << ">>> Starting dense block " << layer->gDenseBlockStart();
        } else if layer->gTranBlockStart() >= 0) {
            if (!prevNl) {
                cout << "\n";
            }
            cout << ">>> Starting tran block " << layer->gTranBlockStart();
        }

        StateBufferAddress inStateSize, outStateSize, totalStateSize;

        if (layer->qStoreInSB()) {
            inStateSize = layer.gInputStateMemWithoutBatching();
            outStateSize = layer.gOutputStateMemWithoutBatching();
            totalStateSize = inStateSize + outStateSize;
            if (totalStateSize > maxStateSize) {
                maxStateSize = totalStateSize;
            }
        } else {
            inStateSize = layer.gInputSize();
            outStateSize = layer.gOutputSize();
        }

        numStr = layer.gNumberStr()
        print (numStr + " " + str(layer))
        layer.m_NumStr = numStr

        prevNl = False;
        if layer.gDenseBlockEnd() >= 0: {
            print("<<< Ending dense block " + str(layer.gDenseBlockEnd()))
            print
            prevNl = True
        } elif layer.gTranBlockEnd() >= 0: {
            print("<<< Ending tran block " + str(layer.gTranBlockEnd()))
            print
            prevNl = True
        }

        self.__PrevLayer =layer
    }

    print("Max state size =", kstr(maxStateSize))
}

//------------------------------------------------
void
Printer::printDot()
{
    Network* ntwk = m_Network;
    FILE* f1 = fopen(netwk.gName()+".dot", 'w')

    string graphName = netwk.gName();
    fprintf(f1, "digraph %s {\n", graphName.c_str());

    for (layer in netwk.gLayers()) {
        string label = layer.gDotIdLabel();
        fprint(f1, "  %s\n", label.c_str());
    }

    fprint(f1, "\n");

    for (layer in netwk.__Layers) {
        for (nextLayer in layer.gNextLayers()) {
            print >>f1, '  ', layer.gDotId(), '->', nextLayer.gDotId(), ';'
        }
    }

    print >>f1, '}'
    print >>f1
}


#-----------------------------------------------------------------
def printLevels(self):
{
        ntwk = self.__Network
        for level in ntwk.gLevels():
            for layer in level.gLayers():
                print(layer.gNameWithSched(),)
            print
}

#-----------------------------------------------------------------
def printSched(self):
{
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
}

def printJsonOld(self, obj, filename):
{
        obj_str = json.dumps(obj.gJson(), sort_keys=False, indent=4, separators=(',', ': '))
        with open(filename, "w") as f:
            f.write(obj_str)
}

def printJson(self, obj, filename):
{
        #obj_str = json.dumps(obj.gJson(), sort_keys=False, indent=4, separators=(',', ': '))
        obj_json = obj.gJson()
        #print("type:", type(obj_json))
        #print(obj_json)
        with open(filename, "w") as f:
            json.dump(obj_json, f, indent=2)
            f.write("\n")
}

