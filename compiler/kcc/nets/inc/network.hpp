#pragma once

#ifndef KCC_NETS_NETWORK_H
#define KCC_NETS_NETWORK_H


#include <string>
#include <vector>
#include <assert.h>

using std::string;
using std::vector;


#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"


namespace kcc {

namespace layers {
    class Layer;
}

namespace nets {

using namespace utils;
using layers::Layer;

//--------------------------------------------------------
class Network {
public:
    static const char* const net_name_key;
    static const char* const data_type_key;
    static const char* const layers_key;
    class SchedForwLayers;
    class SchedRevLayers;
private:
    class SchedLayerForwRevIter;

public:

public:
    //----------------------------------------------------------------
    Network(const DataType& dataType, const string& netName);

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
        return m_DataType;
    }

    void addLayer(Layer* layer);
    
    SchedForwLayers gSchedForwLayers();
    SchedRevLayers gSchedRevLayers();

private:
    const DataType& m_DataType;
    string m_Name;
    vector<Layer*> m_Layers;
    bool m_DoBatching;
}; // class Layer




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
    Layer* m_CurrLayer;
    const bool m_Forw;
};

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
class Network::SchedRevLayers {
public:
    SchedRevLayers(std::vector<Layer*>& layers)
        : m_Layers(layers)
    { }
    SchedLayerForwRevIter begin() const {
        return SchedLayerForwRevIter(m_Layers[m_Layers.size()-1], false);
    }
    SchedLayerForwRevIter end() const {
        return SchedLayerForwRevIter(nullptr, true);
    }
private:
    vector<Layer*>& m_Layers;
};

} // namespace nets
} // namespace kcc

#endif // KCC_NETS_NETWORK_H

