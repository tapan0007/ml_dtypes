
#include "utils/inc/asserter.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

namespace kcc {
namespace kelf {


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlock::DmaBlock(DmaDescription& dmaDescription, const char* comment)
    : m_DmaDescription(dmaDescription)
    , m_Comment(comment)
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
void
DmaDescription::DmaBlock::setDmaEventField(nlohmann::json& jDmaBlock) const
{
    if (m_TailEventIds.size() > 0) {
        if (m_TailEventIds.size() == 1) {
            jDmaBlock["event"]  = m_TailEventIds[0];
        }
        else {
            jDmaBlock["event"]  = m_TailEventIds;
        }
    }
}


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockNonIo::DmaBlockNonIo(DmaDescription& dmaDescription, EngineId engId, const char* comment)
    : DmaBlock(dmaDescription, comment)
    , m_EngineId(engId)
{ }




/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockToTpb::DmaBlockToTpb(DmaDescription& dmaDescription, EngineId engId, bool qWeights, const char* comment)
    : DmaBlockNonIo(dmaDescription, engId, comment)
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
                      m_DmaDescription.gFileSymbolicId(refFile, FileType::Weight));

    m_Descs.push_back(desc);
}


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockFromTpb::DmaBlockFromTpb(DmaDescription& dmaDescription,
        EngineId engId, bool qOut, const char* comment)
    : DmaBlockNonIo(dmaDescription, engId, comment)
    , m_QOut(qOut)
{
    if (qOut) {
        //m_QueueName = gSymbolicOutQueue();
        m_QueueName = m_DmaDescription.gSymbolicQueue(engId, false, false);
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
    const FileIdType idType = m_DmaDescription.gFileSymbolicId(refFile, FileType::LoadSave);
    DmaDescFromTpb desc(numBytes, srcTpbSbAddress, dstFileAddress, idType);

    m_Descs.push_back(desc);
}


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockInput::DmaBlockInput(DmaDescription& dmaDescription,
            EngineId engId, const char* comment)
    : DmaBlock(dmaDescription, comment)
    , m_EngineId(engId)
{
    m_QueueName = m_DmaDescription.gSymbolicQueue(engId, true, false);
    m_BlockId = m_DmaDescription.gBlockIdForQueue(m_QueueName);
}



/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaBlockInput::addDmaDesc(TongaAddress inputAddress,
        TongaAddress dstSbAddress,
        const std::string& refFile,
        kcc_int32 numBytes)
{
    const FileIdType& idType(m_DmaDescription.gInFileSymbolicId(refFile));
    DmaDescToTpb desc(numBytes, dstSbAddress, inputAddress, idType);

    m_Descs.push_back(desc);
}


/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaDescFromTpb::assertAccessCheck() const
{
    const arch::StateBuffer stateBuf(arch::Arch::gArch().gStateBuffer());
    const kcc_uint32 size    = gNumBytes();
    const tpb_addr sbReadAddr  = gSrcSbAddress();
    Assert(stateBuf.qTpbReadAccessCheck(sbReadAddr, size),
        "Unaligned DMA state buffer read access. Addr=",
        std::hex, sbReadAddr, std::dec, " size=", std::dec, size);
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaDescToTpb::assertAccessCheck() const
{
    const arch::StateBuffer stateBuf(arch::Arch::gArch().gStateBuffer());
    const tpb_addr sbWriteAddr = gDstSbAddress(); 
    const kcc_uint32 size   = gNumBytes();
    Assert(stateBuf.qTpbWriteAccessCheck(sbWriteAddr, size),
        "Unaligned DMA state buffer write access. Addr=",
        std::hex, sbWriteAddr, std::dec, " size=", size);
}

}}


