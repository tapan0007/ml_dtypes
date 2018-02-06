#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/waveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

//----------------------------------------------------------------
WaveOp::WaveOp(const WaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : m_Name(params.m_WaveOpName)
    , m_OfmapDesc(params.m_OfmapDesc)
    , m_Layer(params.m_Layer)
{
    for (auto prevWaveOp : prevWaveOps) {
        m_PrevWaveOps.push_back(prevWaveOp);
    }
}

//----------------------------------------------------------------
const utils::DataType&
WaveOp::gDataType() const
{
    return m_Layer->gDataType();
}


bool
WaveOp::verify() const
{
    return true;
}

}} // namespace

