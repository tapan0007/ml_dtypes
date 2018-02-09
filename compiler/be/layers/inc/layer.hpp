#pragma once

#ifndef KCC_LAYERS_LAYER_H
#define KCC_LAYERS_LAYER_H


#include <string>
#include <vector>
#include <assert.h>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"


namespace kcc {

namespace arch {
    class Arch;
}

namespace nets {
    class Network;
}

namespace layers {

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
        const std::vector<Layer*>& prevLayers);

    virtual ~Layer()
    {}

private:
    Layer() = delete;
    Layer(const Layer&) = delete;

    Layer& operator= (const Layer&) const = delete;

protected:
    static std::vector<Layer*> mkLayerVector2(Layer* layer1, Layer* layer2);

    virtual bool verify() const = 0;

public:
    //----------------------------------------------------------------
    virtual std::string gString() const = 0;

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
    virtual bool qConstLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qDataLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qArithmeticLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qAddLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qResAddLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    virtual bool qBiasAddLayer() const {
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
        assert(gEarlyLevel() <= lev && lev <= gLateLevel() && "Layer current level not within early and late level range");
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
    std::vector<Layer*>& gPrevLayers() {
        return m_PrevLayers;
    }

    //----------------------------------------------------------------
    const std::vector<Layer*>& gPrevLayers() const {
        return m_PrevLayers;
    }

    //----------------------------------------------------------------
    Layer* gPrevLayer(kcc_int32 idx) const {
        assert(0 <= idx and idx < gNumPrevLayers() && "Layer: previous layer outside of range");
        return m_PrevLayers[idx];
    }

    //----------------------------------------------------------------
    kcc_int32 gNumPrevLayers() const {
        return m_PrevLayers.size();
    }

    //----------------------------------------------------------------
    std::vector<Layer*>& gNextLayers() {
        return m_NextLayers;
    }

    //----------------------------------------------------------------
    const std::vector<Layer*>& gNextLayers() const {
        return m_NextLayers;
    }

    //----------------------------------------------------------------
    Layer* gNextLayer(kcc_int32 idx) {
        assert(0 <= idx and idx < gNumNextLayers() && "Layer: next layer outside of range");
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
    void rNumberStr(const std::string& numStr) {
        m_NumberStr = numStr;
    }

    //----------------------------------------------------------------
    std::string gNumberStr() const {
        return m_NumberStr;
    }

    //----------------------------------------------------------------
    nets::Network* gNetwork() const {
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

    const std::string gRefFileFormat() const {
        return m_RefFileFormat;
    }

    //----------------------------------------------------------------
    std::string gNameType() const {
        return gName() + "{" + gTypeStr() + "}";
    }

    //----------------------------------------------------------------
    std::string gName() const {
        return m_LayerName;
    }

    void rNumStr(const std::string& s) {
        m_NumStr = s;
    }

    //----------------------------------------------------------------

    std::string gNameNum() const {
        return gName() + "-" + m_NumStr;
    }

    //----------------------------------------------------------------
    std::string gDotId() const {
        std::string numStr = m_NumStr;
        for (auto& ch : numStr) {
            if (ch == '.') {
                ch = '_';
            }
        }
        return gName() + "_" + numStr;
    }

    //----------------------------------------------------------------
    std::string gDotLabel() const {
        std::string s("\"");
        return s + gName() + "-" + m_NumStr + "\"";
    }

    //----------------------------------------------------------------
    std::string gDotIdLabel() const {
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
        assert(m_RefCount >= -num && "Remaining reference count too small");
        m_RefCount += num;
    }

    kcc_int32 gNumPredecessors() const {
        return m_NumPredecessors;
    }
    void changeNumPredecessors(kcc_int32 num) {
        m_NumPredecessors += num;
    }

    //----------------------------------------------------------------
    std::vector<Layer*>& gPrevSbLayers() {
        return m_PrevSbLayers;
    }

    const std::vector<Layer*>& gPrevSbLayers() const {
        return m_PrevSbLayers;
    }

    kcc_int32 gNumPrevSbLayers() {
        return m_PrevSbLayers.size();
    }

    //----------------------------------------------------------------
    std::vector<Layer*>& gNextSbLayers() {
        assert(qStoreInSB());
        return m_NextSbLayers;
    }

    const std::vector<Layer*>& gNextSbLayers() const {
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

    const arch::Arch& gArch() const;

    std::string gBaseLayerStr() const;
    std::string gStateSizesStr() const;
    std::string gNameWithSched() const;
    std::string gNameWithSchedMem() const;

    //----------------------------------------------------------------
    std::string gRefFileName() const {
        return m_RefFileName;
    }

    void rRefFileName(const std::string& refFileName) {
        m_RefFileName = refFileName;
    }



protected:
    std::string             m_LayerName;

    std::vector<Layer*>     m_PrevLayers;
    std::vector<Layer*>     m_NextLayers;
    std::vector<Layer*>     m_PrevSbLayers;
    std::vector<Layer*>     m_NextSbLayers;
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
    std::string             m_NumberStr;
    std::string             m_NumStr;

    FmapDesc                m_OfmapDesc;
private:
    std::string             m_RefFileName;
    std::string             m_RefFileFormat;
}; // class Layer


class Layer::Params {
public:
    bool verify () const;
public:
    std::string             m_LayerName;
    kcc_int32               m_BatchFactor = -1;
    nets::Network*          m_Network = nullptr;
    std::string             m_RefFile;
    std::string             m_RefFileFormat;
};


} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_LAYER_H

