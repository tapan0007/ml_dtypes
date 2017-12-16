#pragma once

#ifndef KCC_NETS_NETWORK_H
#define KCC_NETS_NETWORK_H


#include <string>
#include <vector>
#include <assert.h>

using std::string;
using std::vector;


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
using layers::Layer;
using schedule::LayerLevel;

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
    Layer* findLayer(const string& prevLayerName);

public:
    //----------------------------------------------------------------
    Network()
        : m_DataType(nullptr)
        , m_Name()
        , m_DoBatching(false)
    {}

    Network(const DataType* dataType, const string& netName);

    bool qDoBatching() const {
        return m_DoBatching;
    }
    void rDoBatching(bool doBatch) {
        m_DoBatching = doBatch;
    }

    std::vector<Layer*>& gLayers() {
        return m_Layers;
    }

    Layer* gLayer(int idx) const {
        return m_Layers[idx];
    }

    int gNumberLayers() const {
        return m_Layers.size();
    }

    const DataType& gDataType() const {
        return *m_DataType;
    }

    void addLayer(Layer* layer);

    const string& gName() const {
        return m_Name;
    }

    SchedForwLayers gSchedForwLayers();
    SchedRevLayers gSchedRevLayers();

private:
    const DataType*          m_DataType;
    string                   m_Name;
    vector<Layer*>           m_Layers;
    bool                     m_DoBatching;
    std::map<string, Layer*> m_Name2Layer;
}; // Network




//----------------------------------------------------------------
// Iterates over scheduled layers either forward or in reverse
//----------------------------------------------------------------
class Network::SchedLayerForwRevIter {
public:
    SchedLayerForwRevIter(Layer* startLayer, bool forw)
        : m_CurrLayer(startLayer)
        , m_Forw(forw)
    { }

    bool operator!= (const SchedLayerForwRevIter& rhs) const {
        return m_CurrLayer != rhs.m_CurrLayer;
    }

    Layer* operator* () const {
        return m_CurrLayer;
    }

    void operator++();
private:
    Layer*      m_CurrLayer;
    const bool  m_Forw;
};

//----------------------------------------------------------------
// Iterates over scheduled layers forward
//--------------------------------------------------------
class Network::SchedForwLayers {
public:
    SchedForwLayers(std::vector<Layer*>& layers)
        : m_Layers(layers)
    { }
    SchedLayerForwRevIter begin() const {
        return SchedLayerForwRevIter(m_Layers[0], true);
    }
    SchedLayerForwRevIter end() const {
        return SchedLayerForwRevIter(nullptr, true);
    }
private:
    vector<Layer*>& m_Layers;
};

//--------------------------------------------------------
// Iterates over scheduled layers in reverse
//--------------------------------------------------------
class Network::SchedRevLayers {
public:
    SchedRevLayers(std::vector<Layer*>& layers)
        : m_Layers(layers)
    { }
    SchedLayerForwRevIter begin() const {
        return SchedLayerForwRevIter(m_Layers[m_Layers.size()-1], false);
    }
    SchedLayerForwRevIter end() const {
        return SchedLayerForwRevIter(nullptr, false);
    }
private:
    vector<Layer*>& m_Layers;
};

} // namespace nets
} // namespace kcc

#endif // KCC_NETS_NETWORK_H

