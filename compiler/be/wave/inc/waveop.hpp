#pragma once

#ifndef KCC_WAVE_WAVEOP_H
#define KCC_WAVE_WAVEOP_H


#include <string>
#include <vector>
#include <assert.h>



#include "shared/inc/tpb_isa.hpp"


#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "events/inc/events.hpp"


namespace kcc {

namespace layers {
    class Layer;
}
namespace nets {
    class Network;
}

namespace wave {

class WaveEdge;

using namespace utils;

//--------------------------------------------------------
// The base class of all wave.
//--------------------------------------------------------
class WaveOp { // abstract class
protected:

    //----------------------------------------------------
public:
    class Params;

    enum : kcc_int32 { AtomSize = 1024 };

protected:
    //----------------------------------------------------------------
    WaveOp(const WaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);

    virtual ~WaveOp()
    {}

private:
    WaveOp() = delete;
    WaveOp(const WaveOp&) = delete;

    WaveOp& operator= (const WaveOp&) const = delete;

protected:

    virtual bool verify() const = 0;


public:

    //----------------------------------------------------------------
    virtual bool qMatMulWaveOp() const {
        return false;
    }
    virtual bool qSbAtomFileWaveOp() const {
        return false;
    }
    virtual bool qSbAtomSaveWaveOp() const {
        return false;
    }
    virtual bool qPoolWaveOp() const {
        return false;
    }
    virtual bool qActivationWaveOp() const {
        return false;
    }
    virtual bool qResAddWaveOp() const {
        return false;
    }

    virtual EngineId gEngineId() const = 0;

    virtual std::string gTypeStr() const = 0;

    //----------------------------------------------------------------
    const std::string& gName() const {
        return m_Name;
    }

    const std::string& gLayerName() const;

    const std::vector<WaveEdge*>& gPrevWaveEdges() const {
        return m_PrevWaveEdges;
    }

    const std::vector<WaveEdge*>& gSuccWaveEdges() const {
        return m_SuccWaveEdges;
    }

    kcc_int32 gWaveAtomSize () const {
        return 1024;
    }

    kcc_int32 gOrder() const {
        return m_Order;
    }


protected:
#if 0
#endif

protected:
    std::string             m_Name          = "";
    std::vector<WaveEdge*>  m_PrevWaveEdges;
    std::vector<WaveEdge*>  m_SuccWaveEdges;
    FmapDesc                m_OfmapDesc;
    kcc_int32               m_Order = -1;


private:
    layers::Layer*          m_Layer         = nullptr;
}; // class WaveOp


class WaveOp::Params {
public:
    bool verify() const;
public:
    std::string             m_WaveOpName    = "";
    //FmapDesc                m_OfmapDesc;
    layers::Layer*          m_Layer         = nullptr;
    kcc_int32               m_Order;
};


} // namespace wave
} // namespace kcc

#endif // KCC_WAVE_WAVEOP_H

