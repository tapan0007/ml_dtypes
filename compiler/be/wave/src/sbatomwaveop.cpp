
#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/sbatomwaveop.hpp"
#include "nets/inc/network.hpp"

//#define ASSERT_RETURN(x) return(x)
#define ASSERT_RETURN(x) assert(x); return (x)


namespace kcc {
namespace wave {

SbAtomWaveOp::SbAtomWaveOp(const SbAtomWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_SbAddress(params.m_SbAddress)
    , m_StartAtMidPart(params.m_StartAtMidPart)
    , m_DataType(DataType::dataTypeId2DataType(params.m_DataType))
    , m_Length(params.m_Length)
    , m_OffsetInFile(params.m_OffsetInFile)
    , m_PartitionStepBytes(params.m_PartitionStepBytes)
    , m_RefFileName(params.m_RefFileName)
    , m_RefFileFormat(params.m_RefFileFormat)
    , m_RefFileShape(params.m_RefFileShape)
{
    assert(params.verify());
}


bool
SbAtomWaveOp::verify() const
{
    if (! this-> WaveOp::verify()) {
        ASSERT_RETURN(false);
    }
    if (m_SbAddress < 0) {
        ASSERT_RETURN(false);
    }
    // m_DataType
    if (m_Length <= 0) {
        ASSERT_RETURN(false);
    }
    if (m_OffsetInFile < 0) {
        ASSERT_RETURN(false);
    }
    if (m_PartitionStepBytes < 1) {
        ASSERT_RETURN(false);
    }
    if (m_RefFileName == "") {
        ASSERT_RETURN(false);
    }
    if (m_RefFileFormat == "") {
        ASSERT_RETURN(false);
    }
    if (m_RefFileShape.size() != 4) {
        ASSERT_RETURN(false);
    }
    for (const auto n : m_RefFileShape) {
        if (n < 1) {
            ASSERT_RETURN(false);
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
    if (m_SbAddress < 0) {
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

