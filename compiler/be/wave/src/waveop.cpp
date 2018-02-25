#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/waveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

//----------------------------------------------------------------
WaveOp::WaveOp (const WaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : m_Name(params.m_WaveOpName)
    , m_OfmapDesc(params.m_OfmapDesc)
    , m_Layer(params.m_Layer)
{
    assert(params.verify());
    for (auto prevWaveOp : prevWaveOps) {
        m_PrevWaveOps.push_back(prevWaveOp);
    }
}

//----------------------------------------------------------------
const std::string&
WaveOp::gLayerName () const
{
    return m_Layer->gName();
}

bool
WaveOp::verify () const
{
    if (m_Name == "") {
        return false;
    }
    if (! m_Layer) {
        return false;
    }
    return true;
}





bool
WaveOp::Params::verify() const
{
    if (m_WaveOpName == "") {
        return false;
    }
    if (! m_Layer) {
        return false;
    }
    return true;
}


}} // namespace

