#include <algorithm>

#include "scheduler.hpp"


namespace kcc {
namespace schedule {


//--------------------------------------------------------
void
Scheduler::verifyLevelization()
{
    for (auto level : m_Levels) {
        const int levNum = level->gLevelNum();
        for (auto layer : level->gLayers()) {
            assert(levNum == layer->gEarlyLevel());
            for (auto nextLayer : layer->gNextLayers()) {
                // cannot say anything about layer.Late and nextLayer.Early
                assert(layer->gEarlyLevel() < nextLayer->gEarlyLevel());
                assert(layer->gLateLevel() < nextLayer->gLateLevel());
                assert(layer->gEarlyLevel() <= layer->gLateLevel());
                assert(layer->gEarlyLevel() <= layer->gCurrLevel() &&
                       layer->gCurrLevel() <= layer->gLateLevel());
            }
        }
    }
}

//--------------------------------------------------------
void
Scheduler::calculateLateLevels()
{
    const int lastLevel = m_Levels.size();

    auto revLevels = m_Levels;
    std::reverse(revLevels.begin(), revLevels.end());
    for (auto level : revLevels) {
        for (auto layer : level->gLayers()) {
            int minNextLastLev = lastLevel;
            for (auto nextLayer : layer->gNextLayers()) {
                minNextLastLev = std::min(minNextLastLev, nextLayer->gLateLevel());
            }
            layer->rLateLevel(minNextLastLev - 1);
        }
    }
}

//--------------------------------------------------------
void
Scheduler::Schedule(Network* ntwk)
{
    m_Network = ntwk;
    m_Layers = ntwk->gLayers();

    //calcFanoutBatch()

    levelize();
    assert(m_Levels[0]->gNumberLayers() == 1 && m_Levels[0]->qDataLevel());


    // Move layers with input smaller than output to latest level for the layer
    for (auto layer : m_Layers) {
        if (layer->gEarlyLevel() == layer->gLateLevel()) {
            continue;
        }
        LayerLevel* earlyLevel = m_Levels[layer->gEarlyLevel()];
        LayerLevel* lateLevel = m_Levels[layer->gLateLevel()];
        assert(layer->gCurrLevel() == layer->gEarlyLevel());
        assert(earlyLevel->qContainsLayer(layer));
        assert(earlyLevel->gLevelNum() == layer->gEarlyLevel());
        assert(lateLevel->gLevelNum() == layer->gLateLevel());
        assert(!lateLevel->qContainsLayer(layer));

        if (layer->gInputSize() < layer->gOutputSize()) {
            earlyLevel->remove(layer);
            lateLevel->append(layer);
            layer->rCurrLevel(layer->gLateLevel());
            assert(!earlyLevel->qContainsLayer(layer));
            assert(lateLevel->qContainsLayer(layer));
        }
    }

    // Schedule within level
    m_currSchedule = 0;
    for (auto level : m_Levels) {
        scheduleLevel(level);
    }
    linkSchedLayers();

    calcSbMem();
}

//--------------------------------------------------------
void Scheduler::linkSchedLayers()
{
    for (auto layer : m_Layers) {
        const int mysch1 = layer->gSchedule() + 1;
        for (auto otherLayer : m_Layers) {
            if (mysch1 == otherLayer->gSchedule()) {
                assert(!layer->gNextSchedLayer());
                assert(!otherLayer->gPrevSchedLayer());
                layer->rNextSchedLayer(otherLayer);
                otherLayer->rPrevSchedLayer(layer);
                break;
            }
        }
    }

    Layer* layerWithoutNextSched = nullptr;
    Layer* layerWithoutPrevSched = nullptr;

    for (auto layer : m_Layers) {
        Layer* nextSchedLayer = layer->gNextSchedLayer();
        if (nextSchedLayer) {
            assert(nextSchedLayer->gPrevSchedLayer() == layer);
            assert(layer->gSchedule() + 1 == nextSchedLayer->gSchedule());
        } else {
            assert(!layerWithoutNextSched);
            layerWithoutNextSched = layer;
        }

        Layer* prevSchedLayer = layer->gPrevSchedLayer();
        if (prevSchedLayer) {
            assert(prevSchedLayer->gNextSchedLayer() == layer);
            assert(prevSchedLayer->gSchedule() + 1 == layer->gSchedule());
        } else {
            assert(!layerWithoutPrevSched);
            layerWithoutPrevSched = layer;
        }
    }

    assert(layerWithoutNextSched && layerWithoutPrevSched);
}

//--------------------------------------------------------
void
Scheduler::scheduleLevel(LayerLevel* level)
{
    if (level->gNumberLayers() == 1) {
        for (auto layer : level->gLayers()) {
            layer->rSchedule(m_currSchedule);
            m_currSchedule += 1;
        }
        return;
    }

    // Schedule a multi-layer level
    // If two (or more) layers have the same successor, schedule one after another
    // Schedule the layer with smaller out-state-buffer footprint later
    // Rething this for multi-successor layers
    auto levelCopy = level->gLayers();
    sortLayers(levelCopy);
    for (auto layer : level->gLayers()) {
        layer->rSchedule(m_currSchedule);
        m_currSchedule += 1;
    }
    return;
}


//--------------------------------------------------------
// Less than between layers
static bool compareLayer(Layer* layer1, Layer* layer2)
{
    const int numNext1 = layer1->gNumNextLayers();
    const int numNext2 = layer2->gNumNextLayers();
    if (numNext1 < numNext2) {
        return true;
    } else if (numNext1 > numNext2) {
        return false;
    }

    const auto id1 = layer1->gLayerId();
    const auto id2 = layer2->gLayerId();
    if (id1 < id2) {
        return true;
    } else if (id1 > id2) {
        return false;
    } else {
        const int sz1 = layer1->gInputStateMemWithoutBatching() + layer1->gOutputStateMemWithoutBatching();
        const int sz2 = layer2->gInputStateMemWithoutBatching() + layer2->gOutputStateMemWithoutBatching();
        if (sz1 < sz2) {
            return true;
        } else {
            return false;
        }
    }
}


//--------------------------------------------------------
void
Scheduler::sortLayers(vector<Layer*>& levelCopy)
{
    //--------------------------------------------------------
    std::sort(levelCopy.begin(), levelCopy.end(), compareLayer);
}

//--------------------------------------------------------
// Before processing layer, certain amount of earlier memory is held in SB.
// The memory equals the layer's inputs plus residue (memory held for future
// layers).
//
// After this layer is processed all its inputs could potentially be released,
// except those held for the future (residues). The memory after the layer
// is earlier mem + layer output - inputs that can be released
//
// Residue for this layer is all memory that must be held. It equals:
// 1. Memory held by future layers that is input to this layer
// 2. Memory held by future layers that is NOT input to this layer
// So, if memory before layer is inMem = a + b + c:
// a. input that can be released
// b. input that cannot be released
// c. non-input (that cannot be released because it is held by other future layers)
// Residue = b + c = (a + b + c) - a = inMem - input.that.can.be.released.
//
// Total memory after is
// b + c + layer.out = inMem - input.that.can.be.released + layer.out
//
void
Scheduler::processLayerSbMemForResidueWithoutBatching(Layer* layer)
{
    assert(layer->qStoreInSB());

    // all subsequent layers refer to this layer
    layer->changeRefCount(layer->gNumNextLayers());
    const StateBufferAddress outSize = layer->gOutputStateMemWithoutBatching();
    const StateBufferAddress inMemBefore = gCurrInMem();

    StateBufferAddress inputCanRelease = 0;
    StateBufferAddress inputCannotRelease = 0;
    for (auto inSbLayer : layer->gPrevSbLayers()) {
        assert(layer != inSbLayer && inSbLayer->qStoreInSB());
        inSbLayer->changeRefCount(-1);  // decrease ref count by 1
        const StateBufferAddress inSbMem = inSbLayer->gOutputStateMemWithoutBatching();
        if (inSbLayer->gRefCount() == 0) {
            inputCanRelease += inSbMem;
        } else {
            inputCannotRelease += inSbMem;
        }
    }

    const StateBufferAddress residueMem = inMemBefore - (inputCanRelease);
    const StateBufferAddress inMemAfter = residueMem + outSize;

    layer->rResMemWithoutBatching(residueMem);
    m_CurrInMem = inMemAfter;
}

//--------------------------------------------------------
void
Scheduler::processLayerSbMemForResidueWithBatching(Layer* layer)
{
    assert(layer->qStoreInSB());

    // all subsequent layers refer to this layer
    layer->changeRefCount(layer->gNumNextLayers());
    const StateBufferAddress outSize = layer->gOutputStateMemWithBatching();
    const StateBufferAddress inMemBefore = gCurrInMem();

    StateBufferAddress canRelease = 0;
    StateBufferAddress cannotRelease = 0;

    for (auto inSbLayer : layer->gPrevSbLayers()) {
        assert(layer != inSbLayer && inSbLayer->qStoreInSB());
        inSbLayer->changeRefCount(-1);  // decrease ref count by 1
        const StateBufferAddress inSbMem = inSbLayer->gOutputStateMemWithBatching();
        if (inSbLayer->gRefCount() == 0) {
            canRelease += inSbMem;
        } else {
            cannotRelease += inSbMem;
        }
    }

    const StateBufferAddress inMemAfter = inMemBefore + outSize - canRelease;
    const StateBufferAddress residueMem = inMemBefore - (canRelease + cannotRelease);

    layer->rResMemWithBatching(residueMem);
    m_CurrInMem = inMemAfter;
}

//--------------------------------------------------------
void
Scheduler::calcFanoutBatch()
{
    for (auto layer : m_Network->gLayers()) {
        int maxFanoutBatchFactor = 0;
        int numFanouts = 0;
        for (auto fanoutLayer : layer->gNextLayers()) {
            numFanouts += 1;
            const int fob = fanoutLayer->gBatchFactor();
            if (fob > maxFanoutBatchFactor) {
                maxFanoutBatchFactor = fob;
            }
        }
        assert(numFanouts == 0 || maxFanoutBatchFactor > 0);
        //layer->rMaxFanoutBatchFactor(maxFanoutBatchFactor);
    }
}


//--------------------------------------------------------
void 
Scheduler::addPrevSbLayers(Layer* layer)
{
    for (auto prevLayer : layer->gPrevLayers()) {
        if (prevLayer->qStoreInSB()) {
            layer->addPrevSbLayer(prevLayer);
        } else {
            for (auto sbLayer : prevLayer->gPrevSbLayers()) {
                auto& prevSbLayers(layer->gPrevSbLayers());
                if (std::find(prevSbLayers.begin(), prevSbLayers.end(), sbLayer) == prevSbLayers.end()) {
                    layer->addPrevSbLayer(sbLayer);
                }
            }
        }
    }
}

//--------------------------------------------------------
//  Li: Batch Oi, Output size Oi
//
//  L1  ---L2---L3-----------------|
//       |                         L6
//       +------------L4---L5 ----/
//   L6:   B6*O6
//   L5:   B5*O5     + (B6-B5)*O5
//         self mem    L5 batch
//   L4:   B4*O4     + (B5-B4)*O4 + (B6-B5)*O5
//         self mem    L4 batch
//   L3:   B3*O3     + (B6-B3)*O3
//         self mem    L3 batch
//   L2:   B2*O2     + (B3-B2)*O2 (B6-B3)*O3
//         self mem    L2 batch
//   L1Ba: (B2-B1)*O1 + (B3-B2)*O2
//         L1 batch a
//   L1Bb: (B4-B1)*O1 + (B5-B4)*O4
//         L1 batch b
//   L1:   B1*O1(self mem) + max(L1a, L1b)
//-----------------------------------------------------------------
void
Scheduler::processSbConnectionForBatching(Layer* prevLayer, Layer* nextLayer)
{
    assert(prevLayer->qStoreInSB() && nextLayer->qStoreInSB());
    const int deltaBatch = nextLayer->gBatchFactor() - prevLayer->gBatchFactor();
    assert(deltaBatch >= 0);
    const StateBufferAddress myBatchMem = deltaBatch * prevLayer->gOutputStateMemWithoutBatching();
    const StateBufferAddress batchMem = myBatchMem + nextLayer->gBatchMem();
    if (batchMem > prevLayer->gBatchMem()) {
        prevLayer->rBatchMem(batchMem);
    }
}

//--------------------------------------------------------
void
Scheduler::calcSbMem()
{
    Network* network = m_Network;

    for (auto layer : network->gSchedForwLayers()) {
        addPrevSbLayers(layer);
    }

    // First determine residue without batching
    m_CurrInMem = 0;
    for (auto layer : network->gSchedForwLayers()) {
        if (!layer->qStoreInSB()) {
            continue;
        }
        processLayerSbMemForResidueWithoutBatching(layer);
    }

    // First determing batching
    for (auto layer : network->gSchedRevLayers()) {
        if (!layer->qStoreInSB()) {
            continue;
        }
        for (auto inSbLayer : layer->gPrevSbLayers()) {
            processSbConnectionForBatching(inSbLayer, layer);
        }
    }

    // Second determine residue with batching
    m_CurrInMem = 0L;
    for (auto layer : network->gSchedForwLayers()) {
        if (!layer->qStoreInSB()) {
            continue;
        }
        processLayerSbMemForResidueWithBatching(layer);
    }
}

//--------------------------------------------------------
StateBufferAddress
Scheduler::gCurrInMem() const
{
    return m_CurrInMem;
}

//--------------------------------------------------------
StateBufferAddress
Scheduler::gHighMemWatermark() const
{
    return m_HighMemWatermark;
}

static bool
zeroPred(Layer* layer)
{
    return layer->gNumPrevLayers() == 0;
}

//--------------------------------------------------------
void
Scheduler::levelize()
{
    for (auto layer : m_Layers) {
        layer->rEarlyLevel(-1);
        layer->changeNumPredecessors(-layer->gNumPredecessors()); // zero num predecessors
    }


    std::vector<LayerLevel*> Levels;

    // get layers without predecessors

    size_t currLevelNum = 0; assert(currLevelNum == Levels.size());

    // put all layers from self.__Layers that don't have predecessors into lev0 (
    std::vector<Layer*> lev0;
    copy_if(m_Layers.begin(), m_Layers.end(), lev0.begin(), zeroPred);

    LayerLevel* currLevel = new LayerLevel(currLevelNum, lev0);

    for (auto layer : currLevel->gLayers()) {
        layer->rEarlyLevel(currLevelNum);
    }

    Levels.push_back(currLevel);  // this is level 0
    int numUnprocessedLayers = gNumberLayers() - currLevel->gNumberLayers();

    while (numUnprocessedLayers > 0) {
        const size_t nextLevelNum = currLevelNum + 1; assert(nextLevelNum == Levels.size());
        LayerLevel* nextLevel = new LayerLevel(nextLevelNum, std::vector<Layer*>());
        for (auto currLayer : currLevel->gLayers()) {
            for (auto nextLayer : currLayer->gNextLayers()) {
                nextLayer->changeNumPredecessors(-1);
                if (nextLayer->gNumPredecessors() == 0) {  // all predecessors in previous layers
                    nextLevel->append(nextLayer);
                }
            }
        }

        currLevel = nextLevel;
        currLevelNum = nextLevelNum;
        numUnprocessedLayers -= currLevel->gNumberLayers();
        Levels.push_back(currLevel);
        for (auto layer : currLevel->gLayers()) {
            layer->rEarlyLevel(currLevelNum);
        }
    }


    m_Levels = Levels;
    calculateLateLevels();
    for (auto layer : m_Layers) {
        layer->rCurrLevel(layer->gEarlyLevel());
    }
    verifyLevelization();
}


}}

