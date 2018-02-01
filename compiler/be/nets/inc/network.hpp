#pragma once

#ifndef KCC_NETS_NETWORK_H
#define KCC_NETS_NETWORK_H


#include <string>
#include <vector>
#include <assert.h>



#include "consts.hpp"
#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"


namespace kcc {

namespace layers {
    class Layer;
}
namespace schedule {
    class LayerLevel;
}

namespace nets {

using namespace utils;

//--------------------------------------------------------
// The whole neural net
//--------------------------------------------------------
class Network {
public:
    class SchedForwLayers;
    class SchedRevLayers;
private:
    class SchedLayerForwRevIter;

public:
    template<typename Archive>
    void save(Archive & archive) const;

    template<typename Archive>
    void load(Archive & archive);

private:
    layers::Layer* findLayer(const std::string& prevLayerName);

public:
    //----------------------------------------------------------------
    Network()
        : m_DataType(nullptr)
        , m_Name()
        , m_DoBatching(false)
    {}

#if 0
    Network(const DataType* dataType, const std::string& netName);
#endif

    bool qDoBatching() const {
        return m_DoBatching;
    }
    void rDoBatching(bool doBatch) {
        m_DoBatching = doBatch;
    }

    std::vector<layers::Layer*>& gLayers() {
        return m_Layers;
    }

    layers::Layer* gLayer(kcc_int32 idx) const {
        return m_Layers[idx];
    }

    kcc_int32 gNumberLayers() const {
        return m_Layers.size();
    }

    const DataType& gDataType() const {
        return *m_DataType;
    }

    void addLayer(layers::Layer* layer);

    const std::string& gName() const {
        return m_Name;
    }

    SchedForwLayers gSchedForwLayers() const;
    SchedRevLayers gSchedRevLayers();

private:
    //const DataType*                  m_DataType;
    std::unique_ptr<DataType>        m_DataType;
    std::string                      m_Name;
    std::vector<layers::Layer*>      m_Layers;
    bool                             m_DoBatching;
    std::map<std::string, layers::Layer*> m_Name2Layer;
}; // Network




//----------------------------------------------------------------
// Iterates over scheduled layers either forward or in reverse
//----------------------------------------------------------------
class Network::SchedLayerForwRevIter {
public:
    SchedLayerForwRevIter(layers::Layer* startLayer, bool forw)
        : m_CurrLayer(startLayer)
        , m_Forw(forw)
    { }

    bool operator!= (const SchedLayerForwRevIter& rhs) const {
        return m_CurrLayer != rhs.m_CurrLayer;
    }

    layers::Layer* operator* () const {
        return m_CurrLayer;
    }

    void operator++();
private:
    layers::Layer*      m_CurrLayer;
    const bool  m_Forw;
};

//----------------------------------------------------------------
// Iterates over scheduled layers forward
//--------------------------------------------------------
class Network::SchedForwLayers {
public:
    SchedForwLayers(const std::vector<layers::Layer*>& layers)
        : m_Layers(layers)
    { }
    SchedLayerForwRevIter begin() const {
        return SchedLayerForwRevIter(m_Layers[0], true);
    }
    SchedLayerForwRevIter end() const {
        return SchedLayerForwRevIter(nullptr, true);
    }
private:
    const std::vector<layers::Layer*>& m_Layers;
};

//--------------------------------------------------------
// Iterates over scheduled layers in reverse
//--------------------------------------------------------
class Network::SchedRevLayers {
public:
    SchedRevLayers(std::vector<layers::Layer*>& layers)
        : m_Layers(layers)
    { }
    SchedLayerForwRevIter begin() const {
        return SchedLayerForwRevIter(m_Layers[m_Layers.size()-1], false);
    }
    SchedLayerForwRevIter end() const {
        return SchedLayerForwRevIter(nullptr, false);
    }
private:
    std::vector<layers::Layer*>& m_Layers;
};

} // namespace nets
} // namespace kcc

#endif // KCC_NETS_NETWORK_H

