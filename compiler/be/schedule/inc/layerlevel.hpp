#pragma once

#ifndef KCC_SCHEDULE_LAYERLEVEL_H
#define KCC_SCHEDULE_LAYERLEVEL_H 1

#include <vector>

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"


namespace kcc {
namespace layers {
    class Layer;
}

//using namespace utils;


namespace schedule {

//--------------------------------------------------------
class LayerLevel {
public:
    //--------------------------------------------------------
    LayerLevel(kcc_int32 levelNum, const std::vector<layers::Layer*>& initLayers);

    //--------------------------------------------------------
    void remove(layers::Layer* layer);

    //--------------------------------------------------------
    void append(layers::Layer* layer);

    //--------------------------------------------------------
    kcc_int32 gLevelNum() const {
        return m_LevelNum;
    }

    //--------------------------------------------------------
    std::vector<layers::Layer*>& gLayers() {
        return m_Layers;
    }

    //--------------------------------------------------------
    bool qContainsLayer(layers::Layer* layer) const;

    //--------------------------------------------------------
    kcc_int32 gNumberLayers() const {
        return m_Layers.size();
    }

    //--------------------------------------------------------
    void appendLayer(layers::Layer* layer) {
        m_Layers.push_back(layer);
    }

    //--------------------------------------------------------
    bool qDataLevel() const;

private:
    kcc_int32 m_LevelNum;
    std::vector<layers::Layer*> m_Layers;
};

}}

#endif // KCC_SCHEDULE_LAYERLEVEL_H

