#pragma once

#ifndef KCC_SCHEDULE_LAYERLEVEL_H
#define KCC_SCHEDULE_LAYERLEVEL_H 1

#include <vector>

#include "consts.hpp"


namespace kcc {
namespace layers {
    class Layer;
}

using namespace utils;
using layers::Layer;


namespace schedule {

//--------------------------------------------------------
class LayerLevel {
public:
    //--------------------------------------------------------
    LayerLevel(int levelNum, const std::vector<Layer*>& initLayers);

    //--------------------------------------------------------
    void remove(Layer* layer);

    //--------------------------------------------------------
    void append(Layer* layer);

    //--------------------------------------------------------
    int gLevelNum() const {
        return m_LevelNum;
    }

    //--------------------------------------------------------
    std::vector<Layer*>& gLayers() {
        return m_Layers;
    }

    //--------------------------------------------------------
    bool qContainsLayer(Layer* layer) const;

    //--------------------------------------------------------
    int gNumberLayers() const {
        return m_Layers.size();
    }

    //--------------------------------------------------------
    void appendLayer(Layer* layer) {
        m_Layers.push_back(layer);
    }

    //--------------------------------------------------------
    bool qDataLevel() const;

private:
    int m_LevelNum;
    std::vector<Layer*> m_Layers;
};

}}

#endif // KCC_SCHEDULE_LAYERLEVEL_H

