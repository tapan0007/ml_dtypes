#include <algorithm>

#include "consts.hpp"
#include "layer.hpp"
#include "layerlevel.hpp"

namespace kcc {

namespace schedule {

LayerLevel::LayerLevel(kcc_int32 levelNum, const std::vector<layers::Layer*>& initLayers)
    : m_LevelNum(levelNum)
    , m_Layers(initLayers)
{}


//--------------------------------------------------------
void
LayerLevel::remove(layers::Layer* layer)
{
    //assert(qContainsLayer(layer));
    auto it = find(m_Layers.begin(), m_Layers.end(), layer);
    assert(it != m_Layers.end());
    auto lastIt = m_Layers.end() - 1;
    std::swap(*it, *lastIt);
    m_Layers.pop_back();
}

//--------------------------------------------------------
void
LayerLevel::append(layers::Layer* layer)
{
    m_Layers.push_back(layer);
}

//--------------------------------------------------------
bool
LayerLevel::qDataLevel() const
{
    for (auto layer : m_Layers) {
        if (!layer->qInputLayer()) {
            return false;
        }
    }
    return true;
}

//--------------------------------------------------------
bool
LayerLevel::qContainsLayer(layers::Layer* layer) const
{
    return std::find(m_Layers.begin(), m_Layers.end(), layer) != m_Layers.end();
}

}}

