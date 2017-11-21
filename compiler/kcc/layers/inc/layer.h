#ifndef KCC_LAYERS_LAYER_H
#define KCC_LAYERS_LAYER_H


#include <string>
#include <vector>
#include <assert.h>

using std::string;
using std::vector;


#include "types.h"
#include "datatype.h"
#include "fmapdesc.h"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

using nets::Network;
using namespace utils;

//--------------------------------------------------------
class Layer { // abstract class
protected:
    static const char* const m_LayerNameKey;
    static const char* const m_TypeKey;
    static const char* const m_OfmapKey;
    static const char* const m_NumberOfmapsKey;
    static const char* const m_OfmapWidthKey;
    static const char* const m_OfmapHeightKey;
    static const char* const m_PrevKayersKey;

    //----------------------------------------------------
public:
    class Params;


    //----------------------------------------------------------------
    Layer(const Params& params, const FmapDesc& fmapDesc, const vector<Layer*>& prevLayers);

    //----------------------------------------------------------------
    virtual std::string gTypeStr() const = 0;

    //----------------------------------------------------------------
    virtual bool verify() const = 0;

    //----------------------------------------------------------------
    typedef std::string Json;

    Json gJson();

    static Json combineJson(const Json& j1, const Json&j2);

    //----------------------------------------------------------------
    bool qSubSampleLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qConvLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qPoolLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qMaxPoolLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qAvgPoolLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qOneToOneLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qActivLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qReluLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qBatchNormLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qDataLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qCombineLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qConcatLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qAddLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qSoftMaxLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    bool qFullLayer() const {
        return false;
    }

    //----------------------------------------------------------------
    int32 gBatchFactor() const {
        return m_BatchFactor;
    }

    //----------------------------------------------------------------
    int64 gBatchMem() const {
        return m_BatchMemory;
    }

    //----------------------------------------------------------------
    void rBatchMem(int64 mem) {
        m_BatchMemory = mem;
    }

    //----------------------------------------------------------------
    int64 gResMemWithoutBatching() const {
        return m_ResMemWithoutBatching;
    }

    //----------------------------------------------------------------
    void rResMemWithoutBatching(int64 mem) {
        m_ResMemWithoutBatching = mem;
    }

    //----------------------------------------------------------------
    int64 gResMemWithBatching() const {
        return m_ResMemWithBatching;
    }

    //----------------------------------------------------------------
    void rResMemWithBatching(int64 mem) {
        m_ResMemWithBatching = mem;
    }

    //----------------------------------------------------------------
    int64 gOutputSize() const {
        const int64 wordSize = gDataType().gSizeInBytes();
        const int64 oneBatchSize = (wordSize * gNumOfmaps() * (gOfmapWidth() * gOfmapHeight()));
        return oneBatchSize;
    }

    //----------------------------------------------------------------
    int64 gInputSize() const {
        int64 sz = 0;
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
    int64 gInputStateMemWithoutBatching() const {
        assert(qStoreInSB());
        int64 sz = 0;
        for (auto inSbLayer : m_PrevSbLayers) {
            sz += inSbLayer->gOutputStateMemWithoutBatching();
        }
        return sz;
    }

    //----------------------------------------------------------------
    int64 gOutputStateMemWithoutBatching() const {
        assert(qStoreInSB());
        if (qStoreInSB()) {
            const int64 oneBatchSize = gOutputSize();
            return oneBatchSize;
        } else {
            return 0;
        }
    }

    //----------------------------------------------------------------
    int64 gOutputStateMemWithBatching() const {
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
    int32 gSchedule() const {
        return m_Schedule;
    }

    void rSchedule(int32 sch) {
        m_Schedule = sch;
    }

    //----------------------------------------------------------------
    int32  gCurrLevel() const {
        return m_CurrLevel;
    }

    //----------------------------------------------------------------
    void rCurrLevel(int32 lev) {
        assert(gEarlyLevel() <= lev && lev <= gLateLevel());
        m_CurrLevel = lev;
    }

    //----------------------------------------------------------------
    int32 gEarlyLevel() const {
        return m_EarlyLevel;
    }

    void rEarlyLevel(int32 level) {
        m_EarlyLevel = level;
    }

    //----------------------------------------------------------------
    int32 gLateLevel() const {
        return m_LateLevel;
    }

    void rLateLevel(int32 level) {
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
    Layer* gPrevLayer(int32 idx) const {
        assert(0 <= idx and idx < gNumPrevLayers());
        return m_PrevLayers[idx];
    }

    //----------------------------------------------------------------
    int32 gNumPrevLayers() const {
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
    Layer* gNextLayer(int32 idx) {
        assert(0 <= idx and idx < gNumNextLayers());
        return m_NextLayers[idx];
    }

    //----------------------------------------------------------------
    int32 gNumNextLayers() const {
        return m_NextLayers.size();
    }

    //----------------------------------------------------------------
    void addNextLayer(Layer* nextLayer) {
        m_NextLayers.push_back(nextLayer);
    }

    //----------------------------------------------------------------
    int64 gMaxNextLayerNumberWeights() const {
        int64 maxNumWeights = 0;
        for (auto nextLayer : gNextLayers()) {
            const int64 numWeights = nextLayer->gNumberWeights();
            if (numWeights > maxNumWeights) {
                maxNumWeights = numWeights;
            }
        }
        return maxNumWeights;
    }


    //----------------------------------------------------------------
    int32 gDenseBlockStart() const {
        return m_DenseBlockStart;
    }

    //----------------------------------------------------------------
    void rDenseBlockStart(int32 val) {
        m_DenseBlockStart = val;
    }

    //----------------------------------------------------------------
    int32 gDenseBlockEnd() const {
        return m_DenseBlockEnd;
    }

    //----------------------------------------------------------------
    void rDenseBlockEnd(int32 val) {
        m_DenseBlockEnd = val;
    }

    //----------------------------------------------------------------
    int32 gTranBlockStart() const {
        return m_TranBlockStart;
    }

    //----------------------------------------------------------------
    void rTranBlockStart(int32 val) {
        m_TranBlockStart = val;
    }

    //----------------------------------------------------------------
    int32 gTranBlockEnd() const {
        return m_TranBlockEnd;
    }

    //----------------------------------------------------------------
    void rTranBlockEnd(int32 val) {
        m_TranBlockEnd = val;
    }

    //----------------------------------------------------------------
    // ConvLayer must override these two methods with correct values
    int64 gNumberWeights() const {
        return 0;
    }

    int64 gNumberWeightsPerPartition() const {
        return 0;
    }

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
    int32 gOfmapWidth() const {
        return m_OfmapDesc.gMapWidth();
    }

    //----------------------------------------------------------------
    int32 gOfmapHeight() const {
        return m_OfmapDesc.gMapHeight();
    }


    //----------------------------------------------------------------
    int32 gNumOfmaps() const {
        return m_OfmapDesc.gNumMaps();
    }

    //----------------------------------------------------------------
    string gNameType() const {
        return gName() + "{" + gTypeStr() + "}";
    }

    //----------------------------------------------------------------
    string gName() const {
        return m_LayerName;
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
    int32 gRefCount() const {
        return m_RefCount;
    }

    //----------------------------------------------------------------
    void changeRefCount(int32 num) {
        assert(m_RefCount >= -num);
        m_RefCount += num;
    }

    //----------------------------------------------------------------
    vector<Layer*>& gPrevSbLayers() {
        return m_PrevSbLayers;
    }

    int32 gNumPrevSbLayers() {
        return m_PrevSbLayers.size();
    }

    //----------------------------------------------------------------
    vector<Layer*> gNextSbLayers() {
        assert(qStoreInSB());
        return m_NextSbLayers;
    }

    int32 gNumNextSbLayers() const {
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

    StateBufferAddress gWeightAddress() const {
        return m_WeightAddress;
    }

    void rIfmapAddress(StateBufferAddress address) {
        m_IfmapAddress = address;
    }

    void rOfmapAddress(StateBufferAddress address) {
        m_OfmapAddress = address;
    }

    void rWeightAddress(StateBufferAddress address) {
        m_WeightAddress = address;
    }


    string gBaseLayerStr() const;
    string gStateSizesStr() const;
    string gNameWithSched() const;
    string gNameWithSchedMem() const;



private:
    std::string         m_LayerName;

    vector<Layer*>      m_PrevLayers;
    vector<Layer*>      m_NextLayers;
    vector<Layer*>      m_PrevSbLayers;
    vector<Layer*>      m_NextSbLayers;
    nets::Network*      m_Network;

    Layer*              m_NextSchedLayer;
    Layer*              m_PrevSchedLayer;

    StateBufferAddress  m_IfmapAddress;
    StateBufferAddress  m_OfmapAddress;
    StateBufferAddress  m_WeightAddress;
    int64               m_ResMemWithBatching;
    int64               m_ResMemWithoutBatching;
    int64               m_BatchMemory;
    int32               m_BatchFactor;
    int32               m_Schedule;
    int32               m_CurrLevel;
    int32               m_EarlyLevel;
    int32               m_LateLevel;

    int32               m_DenseBlockStart;
    int32               m_DenseBlockEnd;
    int32               m_TranBlockStart;
    int32               m_TranBlockEnd;
    int32               m_RefCount;

    LayerId             m_Id;
    string              m_NumberStr;
    string              m_NumStr;

    FmapDesc            m_OfmapDesc;
}; // class Layer


class Layer::Params {
public:
    std::string         m_LayerName;
    int32               m_BatchFactor;
    nets::Network*      m_Network;
};


} // namespace layers
} // namespace kcc

#if 0
    #-----------------------------------------------------------------
    virtual string gString() const = 0;

    #-----------------------------------------------------------------
    def combineJson(it):
        x = {}
        for y in it:
            x.update(y)
            #x = { **x, **y }
        return x

    @classmethod
    def gOfmapDescFromJson(klass, layerDict, nn):
        if nn.gUseDimList():
            of = layerDict[Layer.ofmap_key] ##  : [1, self.gNumOfmaps(), self.gOfmapHeight(), self.gOfmapWidth()]
            return OfmapDesc(of[1], (of[2], of[3]) )
        else:
            nOfmaps = layerDict[Layer.number_ofmaps_key]
            ofmapH = layerDict[Layer.ofmap_height_key]
            return OfmapDesc(nOfmaps, (ofmapW, ofmapH))

    @classmethod
    def gLayerNameFromJson(klass, layerDict):
        layerName = layerDict[Layer.layer_name_key]
        return layerName

    @classmethod
    def gPrevLayersFromJson(klass, layerDict, nn):
        prevLayers = []
        prevLayersNames = layerDict[Layer.prev_layers_key]
        for prevLayerName in prevLayersNames:
            prevLayers.append(nn.gLayerByName(prevLayerName))
        return prevLayers

#endif

#endif // KCC_LAYERS_LAYER_H

