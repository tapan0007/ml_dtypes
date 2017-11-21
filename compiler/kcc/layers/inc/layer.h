#ifndef KCC_LAYERS_LAYER_H
#define KCC_LAYERS_LAYER_H


#include <string>
#include <vector>

using std::string;
using std::vector;


#include "types.h"
#include "datatype.h"


namespace kcc {

namespace network {
    class Network;
}
namespace utils {
    class DataType;
    class FmapDesc;
}

namespace layers {

using network::Network;
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
    Layer(const Params& params, FmapDesc& fmapDesc, vector<Layer>& prevLayers);

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
            oneBatchSize = gOutputSize();
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

    int32 rEarlyLevel(int32 level) {
        m_EarlyLevel = level;
    }

    //----------------------------------------------------------------
    int32 gLateLevel() const {
        return m_LateLevel;
    }

    void rLateLevel(int32 level) {
        mLateLevel = level;
    }

    //----------------------------------------------------------------
    vector<Layer>& gPrevLayers() {
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
    vector<Layer>& gNextLayers() {
        return m_NextLayers;
    }

    //----------------------------------------------------------------
    Layer* gNextLayer(int32 idx) {
        assert(0 <= idx and idx < gNumNextLayers())
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
        maxNumWeights = 0
        for (auto nextLayer : gNextLayers()) {
            numWeights = nextLayer->gNumberWeights();
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
        return self.m_TranBlockEnd;
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
        return m_Ofmap_desc;
    }

    //----------------------------------------------------------------
    int32 gOfmapWidth() const {
        return m_Ofmap_desc.gMapWidth();
    }

    //----------------------------------------------------------------
    int32 gOfmapHeight() const {
        return m_Ofmap_desc.gMapHeight();
    }


    //----------------------------------------------------------------
    int32 gNumOfmaps() const {
        return m_Ofmap_desc.gNumMaps();
    }

    //----------------------------------------------------------------
    string gNameType() const {
        return gName() + "{" + self.gTypeStr() + "}";
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
        string numStr = self.m_NumStr;
        for (auto& ch : numStr) {
            if (ch == '.') {
                ch = '_';
            }
        }
        return self.gName() + "_" + numStr;
    }

    //----------------------------------------------------------------
    string gDotLabel() const {
        string s("\"");
        return s + self.gName() + "-" + self.m_NumStr + "\"";

    //----------------------------------------------------------------
    string gDotIdLabel() const [
        return self.gDotId() + " [label=" + self.gDotLabel() + "];";
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
        return self.m_PrevSchedLayer;
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
    vector<Layer>& gPrevSbLayers() {
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
        assert(prevLayer->qStoreInSB())
        m_PrevSbLayers.push_back(prevLayer);
        if (qStoreInSB()) {
            prevLayer.m_NextSbLayers.push_back(this);
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


private:
    std::string         m_LayerName;
    vector<Layer*>&     m_PrevLayers;
    network::Network*   m_Network;

    StateBufferAddress  m_IfmapAddress;
    StateBufferAddress  m_OfmapAddress;
    StateBufferAddress  m_WeightAddress;
    int64               m_ResMemWithBatching;
    int64               m_ResMemWithoutBatching;
    int64               m_BatchMemory;
    int32               m_BatchFactor;
    int32               m_Schedule;
}; // class Layer


class Layer::Params {
public:
    std::string         m_LayerName;
    int32               m_BatchFactor;
    network::Network*   m_Network;
};


} // namespace layers
} // namespace kcc

    #-----------------------------------------------------------------

    def combineJson(self, it):
        x = {}
        for y in it:
            x.update(y)
            #x = { **x, **y }
        return x

    #-----------------------------------------------------------------


    #-----------------------------------------------------------------
    #-----------------------------------------------------------------
    @abstractmethod
    def __str__(self):
        assert(False)

    #-----------------------------------------------------------------



    #-----------------------------------------------------------------
    def gBaseLayerStr(self):
        i = 0
        s = ""
        for prevLayer in self.gPrevLayers():
            ofmap_desc = prevLayer.gOfmapDesc()
            if i == 0:
                s = str(ofmap_desc)
            else:
                s += "+" + str(ofmap_desc)
        s += "-->" + str(self.gOfmapDesc())
        return s

    #-----------------------------------------------------------------
    def gStateSizesStr(self):
        if self.qStoreInSB() :
            nIn= self.gInputStateMemWithoutBatching()
            nOut = self.gOutputStateMemWithoutBatching()
            iState = kstr(nIn)
            oState = kstr(nOut)
            tState = kstr(nIn + nOut)
        else:
            nIn = self.gInputSize()
            iState = "(" + kstr(nIn) + ")"
            nOut = self.gOutputSize()
            oState = "(" + kstr(nOut) + ")"
            tState = "(" + kstr(nIn+nOut) + ")"

        numWeights = self.gNumberWeights()
        nextSchedLayer = self.gNextSchedLayer()
        nextNumWeights = (nextSchedLayer.gNumberWeights() if nextSchedLayer else 0)

        totMem = nIn + nOut + numWeights + nextNumWeights
        return ("  IState=" + (iState)
             + ",  OState=" + (oState)
             + ",  TState=" + (tState)
             + ",  NumWeights=" + kstr(numWeights)
             + ",  TMem=" + kstr(totMem)
                 )


    #-----------------------------------------------------------------
    def gNameWithSched(self):
        layer = self
        return (layer.gName()
              + ' lev=[' + str(layer.gEarlyLevel())
              +        ',' + str(layer.gLateLevel()) + '] '
              + ' sched=' + str(layer.gSchedule())
              )

    #-----------------------------------------------------------------
    def gNameWithSchedMem(self):
        #Str = kstr
        Str = Kstr
        name = self.gNameType()
        if name == "res3d{Add}":
            x = 3
        ofmapStr = (str(self.gNumOfmaps()) + "*" 
                   + str(self.gOfmapWidth()) + "*" + str(self.gOfmapHeigh()) )
        if self.qStoreInSB():
            inMem = self.gInputStateMemWithoutBatching()
            residueMem = self.gResMemWithoutBatching()
            outMem = self.gOutputStateMemWithBatching()
            batchMem = self.gBatchMem()
            batchDelta = "["
            for nextSbLayer in self.gNextSbLayers():
                d = self.gOutputStateMemWithoutBatching() * (nextSbLayer.gBatchFactor() - self.gBatchFactor())
                ##  d = Str(batchMem - nextSbLayer.gBatchMem())
                if batchDelta == "[":
                    batchDelta += Str(d)
                else:
                    batchDelta += "," + Str(d)
            batchDelta += "]"

            s = (SCHED_MEM_FORMAT) % (
                name,
                ofmapStr,
                Str(inMem), Str(outMem),
                Str(residueMem),
                (Str(batchMem) + "[" + str(self.gBatchFactor()) + "]"),
                batchDelta,
                )
        else:
            inMem = self.gInputSize()
            outMem = self.gOutputSize()
            s = (SCHED_MEM_FORMAT) % (
                name,
                ofmapStr,
                Str(inMem), "("+Str(outMem)+")",
                "",  # residueMem
                "",  # batchMem
                "",  # bdelta
                )
        return s


    #-----------------------------------------------------------------
    # Does this layer store data in the SB?
    # That depends on
    # - the sinks of this layer
    # - scheduling of the sinks
    # Therefore, the return value can be determined only after scheduling is finished.
    #-----------------------------------------------------------------
    def qStoreInSB(self):
        return True ## this because Dana requires SB area for each operation,
                    ## cannot feed convolution directly to pooling

        mySched = self.gSchedule()
        assert(mySched != None)
        nextSchedLayer = self.gNextSchedLayer()
        if not nextSchedLayer: ## output
            return True
        elif nextSchedLayer.qConvLayer() : ## to get to convolution
            return True
        else:
            for nextLayer in self.gNextLayers():
                if nextLayer.gSchedule() > mySched + 1:
                    return True

        assert(self.gNumNextLayers() <= 1)
        return False

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

