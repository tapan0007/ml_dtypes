#pragma once

#ifndef KCC_NETS_NETWORK_H
#define KCC_NETS_NETWORK_H


#include <string>
#include <vector>
#include <assert.h>



#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace arch {
    class Arch;
}

namespace layers {
    class Layer;
}

namespace wave {
    class WaveOp;
    class SbAtomWaveOp;
    class SbAtomLoadWaveOp;
    class SbAtomSaveWaveOp;
    class MatMulWaveOp;
    class PoolWaveOp;
    class ActivationWaveOp;
    class ClipByValueWaveOp;
    class ResAddWaveOp;
    class BarrierWaveOp;
    class NopWaveOp;
}
namespace schedule {
    class LayerLevel;
}
namespace serialize {
    class SerWaveOp;
}

namespace nets {

using namespace utils;



constexpr const char* const NetKey_Layers               = "layers";
constexpr const char* const NetKey_WaveOps              = "waveops";
constexpr const char* const NetKey_NetName              = "net_name";
constexpr const char* const NetKey_DataType             = "data_type";
constexpr const char* const NetKey_GitVersion           = "git_version";




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
    class Load;
    class Save;


private:
    layers::Layer* findLayer(const std::string& prevLayerName);
    wave::WaveOp*  findWaveOp(const std::string& prevWaveOpName);

public:
    //----------------------------------------------------------------
    Network(const arch::Arch& arch, const char* gitVersion);

    ~Network();


    const std::string& gGitVersion() const {
        return m_GitVersion;
    }

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


    std::vector<wave::WaveOp*>& gWaveOps() {
        return m_WaveOps;
    }

    const std::vector<wave::WaveOp*>& gWaveOps() const {
        return m_WaveOps;
    }

    wave::WaveOp* gWaveOp(kcc_int32 waveIdx) const {
        return m_WaveOps[waveIdx];
    }

    kcc_int32 gNumberWaveops() const {
        return m_WaveOps.size();
    }


    const DataType& gDataType() const {
        return *m_DataType;
    }
    const DataType& gInDataType() const {
        return gDataType();
    }
    const std::string& gInTensorFormat() const;
    const std::array<kcc_int32, 4>& gInTensorDimensions() const;
    kcc_int32 gInLayerStride() const;

    kcc_int32 gInDataSizeInBytes() const;
    kcc_int32 gOutDataSizeInBytes() const;

    void addLayer(layers::Layer* layer);

    const std::string& gName() const {
        return m_Name;
    }

    SchedForwLayers gSchedForwLayers() const;
    SchedRevLayers gSchedRevLayers();

    void rUseWave (bool useWave) {
        m_UseWave = useWave;
    }

    void replaceWaveops(std::vector<wave::WaveOp*>& newWaveops);
    void revertSavedWaveops();
    void ClearEvents();



private:
    Network() = delete;
    Network(const Network&) = delete;

private:
    const arch::Arch&                       m_Arch;
    std::unique_ptr<DataType>               m_DataType;
    std::string                             m_Name;
    std::string                             m_GitVersion;
    std::vector<layers::Layer*>             m_Layers;
    std::vector<wave::WaveOp*>              m_WaveOps;
    std::vector<wave::WaveOp*>              m_SaveWaveOps;
    bool                                    m_DoBatching;
    std::map<std::string, layers::Layer*>   m_Name2Layer;
    std::map<std::string, wave::WaveOp*>    m_Name2WaveOp;
    bool                                    m_UseWave = false;
    std::unique_ptr<Load>                   m_Load;
    std::unique_ptr<Save>                   m_Save;
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

