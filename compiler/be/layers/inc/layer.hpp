#pragma once

#ifndef KCC_LAYERS_LAYER_H
#define KCC_LAYERS_LAYER_H


#include <string>
#include <vector>
#include <assert.h>



using std::string;
using std::vector;


#include "types.hpp"
#include "consts.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

using nets::Network;
using namespace utils;

//--------------------------------------------------------
// The base class of all layers.
//--------------------------------------------------------
class Layer { // abstract class
protected:

    //----------------------------------------------------
public:
    class Params;

protected:
    //----------------------------------------------------------------
    Layer(const Params& params, const FmapDesc& fmapDesc,
        const string& dataTensorSemantics,
        const vector<Layer*>& prevLayers);

    virtual ~Layer()
    {}

private:
    Layer() = delete;
    Layer(const Layer&) = delete;

    Layer& operator= (const Layer&) const = delete;

protected:
    static vector<Layer*> mkLayerVector2(Layer* layer1, Layer* layer2);

    virtual bool verify() const = 0;

public:
    //----------------------------------------------------------------
    virtual string gString() const = 0;

    //----------------------------------------------------------------
    virtual const char* gTypeStr() const = 0;

    //----------------------------------------------------------------


    //----------------------------------------------------------------
    virtual bool qPassThrough() const {
        return false;
    }


    //----------------------------------------------------------------
    virtual bool qSubSampleLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qConvLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qPoolLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qMaxPoolLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qAvgPoolLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qOneToOneLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qActivLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qReluLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qTanhLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qBatchNormLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qInputLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qCombineLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qConcatLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qAddLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qSoftMaxLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qFullLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    kcc_int32 gBatchFactor() const {
        return m_BatchFactor;
    }

    //----------------------------------------------------------------
    StateBufferAddress gBatchMem() const {
        return m_BatchMemory;
    }

    //----------------------------------------------------------------
    void rBatchMem(StateBufferAddress mem) {
        m_BatchMemory = mem;
    }

    //----------------------------------------------------------------
    StateBufferAddress gResMemWithoutBatching() const {
        return m_ResMemWithoutBatching;
    }

    //----------------------------------------------------------------
    void rResMemWithoutBatching(StateBufferAddress mem) {
        m_ResMemWithoutBatching = mem;
    }

    //----------------------------------------------------------------
    StateBufferAddress gResMemWithBatching() const {
        return m_ResMemWithBatching;
    }

    //----------------------------------------------------------------
    void rResMemWithBatching(StateBufferAddress mem) {
        m_ResMemWithBatching = mem;
    }

    //----------------------------------------------------------------
    StateBufferAddress gOutputSize() const {
        const StateBufferAddress wordSize = gDataType().gSizeInBytes();
        const StateBufferAddress oneBatchSize = (wordSize * gNumOfmaps() * (gOfmapWidth() * gOfmapHeight()));
        return oneBatchSize;
    }

    //----------------------------------------------------------------
    StateBufferAddress gInputSize() const {
        StateBufferAddress sz = 0;
        for (auto inLayer : m_PrevLayers) {
            sz += inLayer->gOutputSize();
        }
        return sz;
    }

    //----------------------------------------------------------------
    const DataType& gDataType() const;

    //----------------------------------------------------------------
    bool  qStoreInSB() const;

    //----------------------------------------------------------------
    StateBufferAddress gInputStateMemWithoutBatching() const {
        assert(qStoreInSB());
        StateBufferAddress sz = 0;
        for (auto inSbLayer : m_PrevSbLayers) {
            sz += inSbLayer->gOutputStateMemWithoutBatching();
        }
        return sz;
    }

    //----------------------------------------------------------------
    StateBufferAddress gOutputStateMemWithoutBatching() const {
        assert(qStoreInSB());
        if (qStoreInSB()) {
            const StateBufferAddress oneBatchSize = gOutputSize();
            return oneBatchSize;
        } else {
            return 0;
        }
    }

    //----------------------------------------------------------------
    StateBufferAddress gOutputStateMemWithBatching() const {
        assert(qStoreInSB());
        return gBatchFactor() * gOutputStateMemWithoutBatching();
    }



    //----------------------------------------------------------------
    LayerId gLayerId() const {
        return m_Id;
    }

    //----------------------------------------------------------------
    void rLayerId(LayerId id) {
        m_Id = id;
    }

    //----------------------------------------------------------------
    kcc_int32 gSchedule() const {
        return m_Schedule;
    }

    void rSchedule(kcc_int32 sch) {
        m_Schedule = sch;
    }

    //----------------------------------------------------------------
    kcc_int32  gCurrLevel() const {
        return m_CurrLevel;
    }

    //----------------------------------------------------------------
    void rCurrLevel(kcc_int32 lev) {
        assert(gEarlyLevel() <= lev && lev <= gLateLevel());
        m_CurrLevel = lev;
    }

    //----------------------------------------------------------------
    kcc_int32 gEarlyLevel() const {
        return m_EarlyLevel;
    }

    void rEarlyLevel(kcc_int32 level) {
        m_EarlyLevel = level;
    }

    //----------------------------------------------------------------
    kcc_int32 gLateLevel() const {
        return m_LateLevel;
    }

    void rLateLevel(kcc_int32 level) {
        m_LateLevel = level;
    }

    //----------------------------------------------------------------
    vector<Layer*>& gPrevLayers() {
        return m_PrevLayers;
    }

    //----------------------------------------------------------------
    const vector<Layer*>& gPrevLayers() const {
        return m_PrevLayers;
    }

    //----------------------------------------------------------------
    Layer* gPrevLayer(kcc_int32 idx) const {
        assert(0 <= idx and idx < gNumPrevLayers());
        return m_PrevLayers[idx];
    }

    //----------------------------------------------------------------
    kcc_int32 gNumPrevLayers() const {
        return m_PrevLayers.size();
    }

    //----------------------------------------------------------------
    vector<Layer*>& gNextLayers() {
        return m_NextLayers;
    }

    //----------------------------------------------------------------
    const vector<Layer*>& gNextLayers() const {
        return m_NextLayers;
    }

    //----------------------------------------------------------------
    Layer* gNextLayer(kcc_int32 idx) {
        assert(0 <= idx and idx < gNumNextLayers());
        return m_NextLayers[idx];
    }

    //----------------------------------------------------------------
    kcc_int32 gNumNextLayers() const {
        return m_NextLayers.size();
    }

    //----------------------------------------------------------------
    void addNextLayer(Layer* nextLayer) {
        m_NextLayers.push_back(nextLayer);
    }

    //----------------------------------------------------------------
    StateBufferAddress gMaxNextLayerNumberWeights() const {
        StateBufferAddress maxNumWeights = 0;
        for (auto nextLayer : gNextLayers()) {
            const StateBufferAddress numWeights = nextLayer->gNumberWeights();
            if (numWeights > maxNumWeights) {
                maxNumWeights = numWeights;
            }
        }
        return maxNumWeights;
    }


    //----------------------------------------------------------------
    // ConvLayer must override this method with correct values
    StateBufferAddress gNumberWeights() const;

    //----------------------------------------------------------------
    // ConvLayer must override this method with correct values
    virtual StateBufferAddress gNumberWeightsPerPartition() const;

    //----------------------------------------------------------------
    void rNumberStr(const string& numStr) {
        m_NumberStr = numStr;
    }

    //----------------------------------------------------------------
    string gNumberStr() const {
        return m_NumberStr;
    }

    //----------------------------------------------------------------
    Network* gNetwork() const {
        return m_Network;
    }

    //----------------------------------------------------------------
    const FmapDesc& gOfmapDesc() const {
        return m_OfmapDesc;
    }

    //----------------------------------------------------------------
    kcc_int32 gOfmapWidth() const {
        return m_OfmapDesc.gMapWidth();
    }

    //----------------------------------------------------------------
    kcc_int32 gOfmapHeight() const {
        return m_OfmapDesc.gMapHeight();
    }


    //----------------------------------------------------------------
    kcc_int32 gNumOfmaps() const {
        return m_OfmapDesc.gNumMaps();
    }

    const string gDataTensorDimSemantics() const {
        return m_DataTensorDimSemantics;
    }

    //----------------------------------------------------------------
    string gNameType() const {
        return gName() + "{" + gTypeStr() + "}";
    }

    //----------------------------------------------------------------
    string gName() const {
        return m_LayerName;
    }

    void rNumStr(const string& s) {
        m_NumStr = s;
    }

    //----------------------------------------------------------------

    string gNameNum() const {
        return gName() + "-" + m_NumStr;
    }

    //----------------------------------------------------------------
    string gDotId() const {
        string numStr = m_NumStr;
        for (auto& ch : numStr) {
            if (ch == '.') {
                ch = '_';
            }
        }
        return gName() + "_" + numStr;
    }

    //----------------------------------------------------------------
    string gDotLabel() const {
        string s("\"");
        return s + gName() + "-" + m_NumStr + "\"";
    }

    //----------------------------------------------------------------
    string gDotIdLabel() const {
        return gDotId() + " [label=" + gDotLabel() + "];";
    }

    //----------------------------------------------------------------
    Layer* gNextSchedLayer() const {
        return m_NextSchedLayer;
    }

    //----------------------------------------------------------------
    void rNextSchedLayer(Layer* nextSchedLayer) {
        m_NextSchedLayer = nextSchedLayer;
    }

    //----------------------------------------------------------------
    Layer* gPrevSchedLayer() const {
        return m_PrevSchedLayer;
    }

    //----------------------------------------------------------------
    void rPrevSchedLayer(Layer* prevSchedLayer) {
        m_PrevSchedLayer = prevSchedLayer;
    }

    //----------------------------------------------------------------
    kcc_int32 gRefCount() const {
        return m_RefCount;
    }

    //----------------------------------------------------------------
    void changeRefCount(kcc_int32 num) {
        assert(m_RefCount >= -num);
        m_RefCount += num;
    }

    kcc_int32 gNumPredecessors() const {
        return m_NumPredecessors;
    }
    void changeNumPredecessors(kcc_int32 num) {
        m_NumPredecessors += num;
    }

    //----------------------------------------------------------------
    vector<Layer*>& gPrevSbLayers() {
        return m_PrevSbLayers;
    }

    const vector<Layer*>& gPrevSbLayers() const {
        return m_PrevSbLayers;
    }

    kcc_int32 gNumPrevSbLayers() {
        return m_PrevSbLayers.size();
    }

    //----------------------------------------------------------------
    vector<Layer*>& gNextSbLayers() {
        assert(qStoreInSB());
        return m_NextSbLayers;
    }

    const vector<Layer*>& gNextSbLayers() const {
        assert(qStoreInSB());
        return m_NextSbLayers;
    }

    kcc_int32 gNumNextSbLayers() const {
        return m_NextSbLayers.size();
    }

    //----------------------------------------------------------------
    void addPrevSbLayer(Layer* prevLayer) {
        assert(prevLayer->qStoreInSB());
        m_PrevSbLayers.push_back(prevLayer);
        if (qStoreInSB()) {
            prevLayer->m_NextSbLayers.push_back(this);
        }
    }

    StateBufferAddress gIfmapAddress() const {
        return m_IfmapAddress;
    }

    StateBufferAddress gOfmapAddress() const {
        return m_OfmapAddress;
    }

    void rIfmapAddress(StateBufferAddress address) {
        m_IfmapAddress = address;
    }

    void rOfmapAddress(StateBufferAddress address) {
        m_OfmapAddress = address;
    }


    string gBaseLayerStr() const;
    string gStateSizesStr() const;
    string gNameWithSched() const;
    string gNameWithSchedMem() const;

    //----------------------------------------------------------------
    const string gRefFileName() const {
        return m_RefFileName;
    }

    void rRefFileName(const string& refFile) {
        m_RefFileName = refFile;
    }


protected:
    std::string             m_LayerName;

    vector<Layer*>          m_PrevLayers;
    vector<Layer*>          m_NextLayers;
    vector<Layer*>          m_PrevSbLayers;
    vector<Layer*>          m_NextSbLayers;
    nets::Network* const    m_Network;

    Layer*                  m_NextSchedLayer;
    Layer*                  m_PrevSchedLayer;

    StateBufferAddress      m_IfmapAddress;
    StateBufferAddress      m_OfmapAddress;
    StateBufferAddress      m_ResMemWithBatching;
    StateBufferAddress      m_ResMemWithoutBatching;
    StateBufferAddress      m_BatchMemory;
    kcc_int32               m_BatchFactor;
    kcc_int32               m_Schedule;
    kcc_int32               m_CurrLevel;
    kcc_int32               m_EarlyLevel;
    kcc_int32               m_LateLevel;

    kcc_int32               m_RefCount;
    kcc_int32               m_NumPredecessors;

    LayerId                 m_Id;
    string                  m_NumberStr;
    string                  m_NumStr;

    FmapDesc                m_OfmapDesc;
    string                  m_DataTensorDimSemantics;
    string                  m_RefFileName;
}; // class Layer


class Layer::Params {
public:
    std::string             m_LayerName;
    kcc_int32               m_BatchFactor = 1;
    nets::Network*          m_Network = nullptr;
    std::string             m_RefFile;
};


} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_LAYER_H

