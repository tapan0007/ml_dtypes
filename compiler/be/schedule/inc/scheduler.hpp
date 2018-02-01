#pragma once

#ifndef KCC_SCHEDULE_SCHEDULER_H
#define KCC_SCHEDULE_SCHEDULER_H 1

#include <vector>

#include "consts.hpp"
#include "types.hpp"


namespace kcc {
namespace nets {
    class Network;
}
namespace layers {
    class Layer;
}

using namespace utils;

namespace schedule {

class LayerLevel;

//--------------------------------------------------------
class Scheduler {
public:
    //--------------------------------------------------------
    Scheduler()
    { }

public:
    //--------------------------------------------------------
    void Schedule(nets::Network* ntwk);

    //--------------------------------------------------------
    std::vector<LayerLevel*>& gLevels() {
        return m_Levels;
    }

private:
    //--------------------------------------------------------
    kcc_int32 gNumberLayers() const {
        return m_Layers.size();
    }

private:
    //--------------------------------------------------------
    // Level[i] = layers without predecessors in:
    //      All-Layers - Union{k : k in [0,i) : Level[k]}
    //
    // All layers go to the earliest level
    //--------------------------------------------------------
    void levelize();

    //--------------------------------------------------------
    void verifyLevelization();

    //--------------------------------------------------------
    void calculateLateLevels();

private:
    //--------------------------------------------------------
    void linkSchedLayers();

    //--------------------------------------------------------
    void scheduleLevel(LayerLevel* level);

    //--------------------------------------------------------
    void sortLayers(std::vector<layers::Layer*>& levelCopy);

    //--------------------------------------------------------
    void processLayerSbMemForResidueWithoutBatching(layers::Layer* layer);

    //--------------------------------------------------------
    void processLayerSbMemForResidueWithBatching(layers::Layer* layer);

    //--------------------------------------------------------
    void  calcFanoutBatch();

    //--------------------------------------------------------
    void addPrevSbLayers(layers::Layer* layer);

    //-----------------------------------------------------------------
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
    void processSbConnectionForBatching(layers::Layer* prevLayer, layers::Layer* nextLayer);

    //--------------------------------------------------------
    void calcSbMem();

    //--------------------------------------------------------
    StateBufferAddress gCurrInMem() const;

    //--------------------------------------------------------
    StateBufferAddress gHighMemWatermark() const;

private:
    std::vector<layers::Layer*> m_Layers;
    std::vector<LayerLevel*> m_Levels;
    nets::Network* m_Network;
    kcc_int32 m_currSchedule;
    StateBufferAddress m_CurrInMem;
    StateBufferAddress m_HighMemWatermark;
};

}}

#endif // KCC_SCHEDULE_SCHEDULER_H

