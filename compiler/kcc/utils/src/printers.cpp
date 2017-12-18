//import json

#include <iostream>
#include <sstream>

//from utils.funcs     import kstr
#include "consts.hpp"
#include "datatype.hpp"
#include "layer.hpp"
#include "network.hpp"
#include "layerlevel.hpp"
#include "printers.hpp"

namespace kcc {
namespace utils {

//--------------------------------------------------------
void
Printer::printNetwork()
{
    Network* ntwk = m_Network;
    bool prevNl = false;
    StateBufferAddress maxStateSize = 0;
    //kcc_int32 layerNumMajor = 0;
    //kcc_int32 layerNumMinor = 0;
    m_PrevLayer = nullptr;

    for (auto layer : ntwk->gLayers()) {
        if (layer->gDenseBlockStart() >= 0) {
            if (!prevNl) {
                std::cout << "\n";
            }
            std::cout << ">>> Starting dense block " << layer->gDenseBlockStart();
        } else if (layer->gTranBlockStart() >= 0) {
            if (!prevNl) {
                std::cout << "\n";
            }
            std::cout << ">>> Starting tran block " << layer->gTranBlockStart();
        }

        StateBufferAddress inStateSize, outStateSize, totalStateSize;

        if (layer->qStoreInSB()) {
            inStateSize = layer->gInputStateMemWithoutBatching();
            outStateSize = layer->gOutputStateMemWithoutBatching();
            totalStateSize = inStateSize + outStateSize;
            if (totalStateSize > maxStateSize) {
                maxStateSize = totalStateSize;
            }
        } else {
            inStateSize = layer->gInputSize();
            outStateSize = layer->gOutputSize();
        }

        string numStr(layer->gNumberStr());
        std::cout << numStr << " " << layer->gString();
        layer->rNumStr(numStr);

        prevNl = false;
        if (layer->gDenseBlockEnd() >= 0) {
            std::cout << "<<< Ending dense block " << layer->gDenseBlockEnd();
            std::cout << "\n";
            prevNl = true;
        } else if (layer->gTranBlockEnd() >= 0) {
            std::cout << "<<< Ending tran block " << layer->gTranBlockEnd();
            std::cout << "\n";
            prevNl = true;
        }

        m_PrevLayer = layer;
    }

    std::cout << "Max state size =" << maxStateSize;
}

//------------------------------------------------
void
Printer::printDot()
{
    Network* ntwk = m_Network;
    const string dotFileName(ntwk->gName() + ".dot");
    FILE* f1 = fopen(dotFileName.c_str(), "w");

    string graphName = ntwk->gName();
    for (auto& ch : graphName) {
        if (ch == '-' || ch == '.') {
            ch = '_';
        }
    }
    fprintf(f1, "digraph %s {\n", graphName.c_str());

    for (auto layer : ntwk->gLayers()) {
        fprintf(f1, "  %s\n", layer->gDotIdLabel().c_str());
    }

    fprintf(f1, "\n");

    for (auto layer : ntwk->gLayers()) {
        for (auto nextLayer : layer->gNextLayers()) {
            fprintf(f1, "  %s->%s;\n", layer->gDotId().c_str(), nextLayer->gDotId().c_str());
        }
    }

    fprintf(f1, "}\n\n");
}


//-----------------------------------------------------------------
void
Printer::printLevels()
{
    Network* ntwk = m_Network;
    for (auto level : ntwk->gLevels()) {
        for (auto layer : level->gLayers()) {
            std::cout << (layer->gNameWithSched()) << "\n";
        }
        std::cout << "\n";
    }
}

//-----------------------------------------------------------------
void
Printer::printSched()
{
    Network* ntwk = m_Network;
    const DataType& dataType(ntwk->gDataType());
    std::cout << ntwk->gName() << ": data type=" << dataType.gName()
              << " data type size=" << dataType.gSizeInBytes() << "\n";
    char memHeader[256];
    sprintf(memHeader, SCHED_MEM_FORMAT,
        "Layer", "Ofmap", "In", "Out",
        "Residue", "Batch", "BatchDlt");
    const char* lineFmt = "%-70s  %s";
    char fullHeader[256];
    sprintf(fullHeader, lineFmt, memHeader, "SB predecessors");
    std::cout << fullHeader;
    bool hasRelu = false;
    bool lastWasAdd = false;

    for (auto layer : ntwk->gSchedForwLayers()) {
        if (layer->qReluLayer()) {
            hasRelu = true;
        }
        string sbPreds = "";
        bool first=true;
        for (auto sbLayer : layer->gPrevSbLayers()) {
            string s = sbLayer->gName();
            if (! first) {
                s = "," + s;
            }
            first=false;
            sbPreds += s;
        }

        if (sbPreds == "") {
            sbPreds = "()";
        }
        const char* sb = layer->qStoreInSB() ? "SB" : "--";
        char line[256];
        sprintf(line, lineFmt, layer->gNameWithSchedMem().c_str(), sb, sbPreds.c_str());
        std::stringstream ss;
        ss << line;

        StateBufferAddress ifaddr = layer->gIfmapAddress();
        StateBufferAddress ofaddr = layer->gOfmapAddress();
        StateBufferAddress waddr  = layer->gWeightAddress();

        if (ifaddr != StateBufferAddress_Invalid || ofaddr != StateBufferAddress_Invalid ||
           waddr != StateBufferAddress_Invalid)
        {
            ss << " {";
            bool b = false;
            if (ifaddr != StateBufferAddress_Invalid) {
                ss << "i=" << ifaddr;
                b = true;
            }
            if (ofaddr != StateBufferAddress_Invalid) {
                if (b) {
                    ss << ", ";
                }
                ss << "o=" << ofaddr;
                b = true;
            }
            if (waddr != StateBufferAddress_Invalid) {
                if (b) {
                    ss << ", ";
                }
                ss << "w=" << waddr;
                b = true;
            }
            ss << "}";
        }

        std::cout << ss.str() << "\n";
        if (hasRelu) {
            if (lastWasAdd && layer->qReluLayer()) {
                std::cout << "\n";
            }
        } else {
            if (layer->qAddLayer() || layer->qPoolLayer()) {
                std::cout << "\n";
            }
        }

        lastWasAdd = layer->qAddLayer();

    }
    std::cout << fullHeader << "\n";
}


}}


