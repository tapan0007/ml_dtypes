#pragma once

#ifndef KCC_SCHEDULE_LAYERLEVEL_H
#define KCC_SCHEDULE_LAYERLEVEL_H 1

#include <vector>

#include "consts.hpp"
#include "types.hpp"


namespace kcc {
namespace layers {
    class Layer;
}

//using namespace utils;
using layers::Layer;


namespace schedule {

//--------------------------------------------------------
class LayerLevel {
public:
    //--------------------------------------------------------
    LayerLevel(kcc_int32 levelNum, const std::vector<Layer*>& initLayers);

    //--------------------------------------------------------
    void remove(Layer* layer);

    //--------------------------------------------------------
    void append(Layer* layer);

    //--------------------------------------------------------
    kcc_int32 gLevelNum() const {
        return m_LevelNum;
    }

    //--------------------------------------------------------
    std::vector<Layer*>& gLayers() {
        return m_Layers;
    }

    //--------------------------------------------------------
    bool qContainsLayer(Layer* layer) const;

    //--------------------------------------------------------
    kcc_int32 gNumberLayers() const {
        return m_Layers.size();
    }

    //--------------------------------------------------------
    void appendLayer(Layer* layer) {
        m_Layers.push_back(layer);
    }

    //--------------------------------------------------------
    bool qDataLevel() const;

private:
    kcc_int32 m_LevelNum;
    std::vector<Layer*> m_Layers;
};

}}

#endif // KCC_SCHEDULE_LAYERLEVEL_H

