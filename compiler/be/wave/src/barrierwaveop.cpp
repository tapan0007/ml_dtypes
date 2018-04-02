#include <sstream>



#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/barrierwaveop.hpp"


// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

BarrierWaveOp::BarrierWaveOp(const WaveOp::Params& params,
                             const std::vector<WaveOp*>& prevWaveOps,
                             const std::vector<WaveOp*>& succWaveOps,
                             EngineId engineId)
    : WaveOp(params, prevWaveOps) // will add back edges
    , m_EngineId(engineId)
{
    Assert(prevWaveOps.size() > 0, "Number of predecessors of barrier waveop ",
            params.m_WaveOpName, " must be >0");
    Assert(succWaveOps.size() > 0, "Number of successors of barrier waveop ",
            params.m_WaveOpName, " must be >0");

    for (WaveOp* succWaveOp : succWaveOps) {
        auto edge = new WaveEdge(this, succWaveOp);
        this->m_SuccWaveEdges.push_back(edge);
        succWaveOp->addPrevWaveEdge(edge);
        succWaveOp->setHasInBarrier();
    }
}

const std::string&
BarrierWaveOp::gLayerName() const
{
    const static std::string name("BarrierLayer");
    return  name;
}



#define RETURN_ASSERT(x)  assert(x); return (x)


bool
BarrierWaveOp::verify() const
{
    // Don't call WaveOp::verify() since the layer is nullptr
    if (m_Name == "") {
        RETURN_ASSERT(false);
    }
    if (m_Order < 0) {
        RETURN_ASSERT(false);
    }
    return true;
}




bool
BarrierWaveOp::Params::verify() const
{
    // Don't call WaveOp::Params::verify() since the layer is nullptr
    if (m_WaveOpName == "") {
        RETURN_ASSERT(false);
    }
    if (m_Order < 0) {
        RETURN_ASSERT(false);
    }
    return true;
}

}}

