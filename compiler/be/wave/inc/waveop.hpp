#pragma once

#ifndef KCC_WAVE_WAVEOP_H
#define KCC_WAVE_WAVEOP_H


#include <string>
#include <vector>
#include <assert.h>





#include "utils/inc/asserter.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "events/inc/events.hpp"


namespace kcc {

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
    virtual bool qSbAtomLoadWaveOp() const {
        return false;
    }
    virtual bool qSbAtomSaveWaveOp() const {
        return false;
    }
    bool qSbAtomWaveOp() const {
        return qSbAtomLoadWaveOp() || qSbAtomSaveWaveOp();
    }
    bool qTpbWaveOp() const {
        return ! qSbAtomWaveOp();
    }
    virtual bool qPoolWaveOp() const {
        return false;
    }
    virtual bool qReciprocalWaveOp() const {
        return false;
    }
    virtual bool qScaleAddWaveOp() const {
        return false;
    }
    virtual bool qActivationWaveOp() const {
        return false;
    }
    virtual bool qTensorTensorWaveOp() const {
        return false;
    }
    virtual bool qTensorScalarConstWaveOp() const {
        return false;
    }
    virtual bool qBarrierWaveOp() const {
        return false;
    }
    virtual bool qNopWaveOp() const {
        return false;
    }

    virtual EngineId gEngineId() const = 0;

    virtual std::string gTypeStr() const = 0;

    virtual WaveOpType gType() const = 0;

    //----------------------------------------------------------------
    const std::string& gName() const {
        return m_Name;
    }

    virtual const std::string& gLayerName() const;

    const std::vector<WaveEdge*>& gPrevWaveEdges() const {
        return m_PrevWaveEdges;
    }

    const std::vector<WaveEdge*>& gSuccWaveEdges() const {
        return m_SuccWaveEdges;
    }

    void addPrevWaveEdge(WaveEdge* waveEdge) {
        m_PrevWaveEdges.push_back(waveEdge);
    }

    void addSuccWaveEdge(WaveEdge* waveEdge) {
        m_SuccWaveEdges.push_back(waveEdge);
    }

    kcc_int32 gNumberPrevWaitEdges() const;

    kcc_int32 gNumberSuccWaitEdges() const;

    kcc_int32 gWaveAtomSize () const {
        return 1024;
    }

    kcc_int32 gOrder() const {
        return m_Order;
    }

    void rOrder(kcc_int32 ord) {
        m_Order = ord;
    }


    bool qHasInBarrier() const {
        return m_HasInBarrier;
    }
    void setHasInBarrier() {
        Assert(! m_HasInBarrier, "Setting in-barrier on waveop that already has in-barrier");
        m_HasInBarrier = true;
    }

    bool qHasOutBarrier() const {
        return m_HasOutBarrier;
    }
    void setHasOutBarrier() {
        Assert(! m_HasOutBarrier, "Setting out-barrier on waveop that already has out-barrier");
        m_HasOutBarrier = true;
    }

protected:
    std::string             m_Name              = "";
    std::vector<WaveEdge*>  m_PrevWaveEdges;
    std::vector<WaveEdge*>  m_SuccWaveEdges;
    FmapDesc                m_OfmapDesc;
    kcc_int32               m_Order             = -1;
    bool                    m_HasInBarrier  = false;
    bool                    m_HasOutBarrier = false;
    std::string             m_LayerName     = "";
}; // class WaveOp


class WaveOp::Params {
public:
    bool verify() const;
public:
    std::string             m_WaveOpName    = "";
    //FmapDesc                m_OfmapDesc;
    std::string             m_LayerName     = "";
    kcc_int32               m_Order;
};


} // namespace wave
} // namespace kcc

#endif // KCC_WAVE_WAVEOP_H

