#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/waveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

//----------------------------------------------------------------
WaveOp::WaveOp(const Params& params,
        const FmapDesc&ofmap_desc,
        const std::vector<WaveOp*>& prevWaveOps)
    : m_Name(params.m_Name)
    , m_OfmapDesc(ofmap_desc)
    , m_Layer(params.m_Layer)
    , m_RefFileName(params.m_RefFile)
    , m_RefFileFormat(params.m_RefFileFormat)
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

