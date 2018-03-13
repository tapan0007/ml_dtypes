#pragma once

#ifndef KCC_WAVE_WAVEEDGE_H
#define KCC_WAVE_WAVEEDGE_H


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
enum class EventWaitMode;
enum class EventSetMode;

namespace layers {
    class Layer;
}
namespace nets {
    class Network;
}

namespace wave {

class WaveOp;

using namespace utils;

//--------------------------------------------------------
// The base class of all wave.
//--------------------------------------------------------
class WaveEdge {
protected:

    //----------------------------------------------------
public:
    //----------------------------------------------------------------
    WaveEdge(WaveOp* fromOp, WaveOp* toOp);

private:
    WaveEdge() = delete;
    WaveEdge(const WaveEdge&) = delete;
    WaveEdge& operator= (const WaveEdge&) const = delete;


public:

    WaveOp* gFromOp() {
        return m_FromOp;
    }

    WaveOp* gToOp() {
        return m_ToOp;
    }

    /****************************************************************
     *                                                              *
     ****************************************************************/
    events::EventId gEventId() const {
        return m_EventId;
    }

private:
    WaveOp*                 m_From;
    WaveOp*                 m_To;
    events::EventId                 m_EventId;
}; // class WaveEdge


class WaveEdge::Params {
public:
    bool verify() const;
public:
    std::string             m_WaveEdgeName    = "";
    //FmapDesc                m_OfmapDesc;
    layers::Layer*          m_Layer         = nullptr;
    kcc_int32               m_Order;
};


} // namespace wave
} // namespace kcc

#endif // KCC_WAVE_WAVEEDGE_H

