

#include "kelf/inc/kelfdmadescription.hpp"

namespace kcc {
namespace kelf {


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlock::DmaBlock(DmaDescription& dmaDescription)
    : m_DmaDescription(dmaDescription)
{
}


/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaBlock::addTailEventId(events::EventId eventId)
{
    m_TailEventIds.push_back(eventId);
}


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockNonIo::DmaBlockNonIo(DmaDescription& dmaDescription, EngineId engId)
    : DmaBlock(dmaDescription)
    , m_EngineId(engId)
{ }




/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockToTpb::DmaBlockToTpb(DmaDescription& dmaDescription, EngineId engId, bool qWeights)
    : DmaBlockNonIo(dmaDescription, engId)
    , m_QWeights(qWeights)
{
    m_QueueName = m_DmaDescription.gSymbolicQueue(engId, true, qWeights);
    m_BlockId = m_DmaDescription.gBlockIdForQueue(m_QueueName);
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaBlockToTpb::addDmaDesc(TongaAddress srcFileAddress,
        const std::string& refFile,
        TpbAddress dstTpbSbAddress, kcc_int32 numBytes)
{
    DmaDescToTpb desc(numBytes, dstTpbSbAddress, srcFileAddress,
                      m_DmaDescription.gFileSymbolicId(refFile));

    m_Descs.push_back(desc);
}


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockFromTpb::DmaBlockFromTpb(DmaDescription& dmaDescription,
        EngineId engId, bool qOut)
    : DmaBlockNonIo(dmaDescription, engId)
    , m_QOut(qOut)
{
    if (qOut) {
        m_QueueName = gSymbolicOutQueue();
    } else {
        m_QueueName = m_DmaDescription.gSymbolicQueue(engId, false, false);
    }
    m_BlockId = m_DmaDescription.gBlockIdForQueue(m_QueueName);
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaBlockFromTpb::addDmaDesc(TpbAddress srcTpbSbAddress,
        TongaAddress dstFileAddress,
        const std::string& refFile,
        kcc_int32 numBytes)
{
    DmaDescFromTpb desc(numBytes, srcTpbSbAddress, dstFileAddress,
        m_QOut ? m_DmaDescription.gSymbolicOutput()
               : m_DmaDescription.gFileSymbolicId(refFile));

    m_Descs.push_back(desc);
}


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockInput::DmaBlockInput(DmaDescription& dmaDescription)
    : DmaBlock(dmaDescription)
{
    m_QueueName = m_DmaDescription.gSymbolicInQueue();
    m_BlockId = m_DmaDescription.gBlockIdForQueue(m_QueueName);
}


/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaBlockInput::addDmaDesc(TongaAddress inputAddress,
        TongaAddress dstSbAddress, kcc_int32 numBytes)
{
    DmaDescToTpb desc(numBytes, dstSbAddress, inputAddress, gSymbolicInput());

    m_Descs.push_back(desc);
}


}}


