
#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/sbatomwaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

SbAtomWaveOp::SbAtomWaveOp(const SbAtomWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_AtomId(params.m_AtomId)
    , m_AtomSize(params.m_AtomSize)
    , m_BatchFoldIdx(params.m_BatchFoldIdx)
    , m_Length(params.m_Length)
    , m_OffsetInFile(params.m_OffsetInFile)
    , m_RefFileName(params.m_RefFileName)
    , m_RefFileFormat(params.m_RefFileFormat)
    , m_RefFileShape(params.m_RefFileShape)
{
    assert(params.verify());
    if (params.m_DataType == utils::DataTypeUint8::gNameStatic()) {
        m_DataType = std::make_unique<utils::DataTypeUint8>();
    } else if (params.m_DataType == utils::DataTypeUint16::gNameStatic()) {
        m_DataType = std::make_unique<utils::DataTypeUint16>();
    } else if (params.m_DataType == utils::DataTypeFloat16::gNameStatic()) {
        m_DataType = std::make_unique<utils::DataTypeFloat16>();
    } else if (params.m_DataType == utils::DataTypeFloat32::gNameStatic()) {
        m_DataType = std::make_unique<utils::DataTypeFloat32>();
    } else {
        assert(false && "Wrong data type when (de)serializing");
    }
}


bool 
SbAtomWaveOp::verify() const
{
    if (! this-> WaveOp::verify()) {
        return false;
    }
    if (m_AtomId < 0) {
        return false;
    }
    if (m_AtomSize < 1) {
        return false;
    }
    if (m_BatchFoldIdx < 0) {
        return false;
    }
    if (! m_DataType) {
        return false;
    }
    if (m_Length <= 0) {
        return false;
    }
    if (m_OffsetInFile < 0) {
        return false;
    }
    if (m_RefFileName == "") {
        return false;
    }
    if (m_RefFileFormat == "") {
        return false;
    }
    if (m_RefFileShape.size() != 4) {
        return false;
    }
    for (const auto n : m_RefFileShape) {
        if (n < 1) {
            return false;
        }
    }
    return true;
}



bool 
SbAtomWaveOp::Params::verify() const
{
    if (! this-> WaveOp::Params::verify()) {
        return false;
    }
    if (m_RefFileName == "") {
        return false;
    }
    if (m_BatchFoldIdx < 0) {
        return false;
    }
    if (m_AtomId < 0) {
        return false;
    }
    if (m_Length < 0) {  // TODO: should be <=
        return false;
    }
    if (m_OffsetInFile < 0) {
        return false;
    }
    return true;
}


}}

