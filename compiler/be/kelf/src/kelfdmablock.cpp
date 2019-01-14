
#include "utils/inc/asserter.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

namespace kcc {
namespace kelf {


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlock::DmaBlock(DmaDescription* dmaDescription,
                        const dma::DmaQueue* que, const char* comment)
    : m_DmaDescription(dmaDescription)
    , m_Queue(que)
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
DmaDescription::DmaBlockNonIo::DmaBlockNonIo(DmaDescription* dmaDescription,
            const dma::DmaQueue* que, EngineId engId, const char* comment)
    : DmaBlock(dmaDescription, que, comment)
    , m_EngineId(engId)
{ }




/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockToTpb::DmaBlockToTpb(DmaDescription* dmaDescription,
            const dma::DmaQueue* que, EngineId engId, bool qWeights, const char* comment)
    : DmaBlockNonIo(dmaDescription, que, engId, comment)
    , m_QWeights(qWeights)
{
    if (qWeights) {
        Assert(que->gQueueType() == dma::DmaQueue::QueueType::Weights,
               "DMA block to TPB with weights must use weight queue");
    } else {
        Assert(que->gQueueType() == dma::DmaQueue::QueueType::Input
               || que->gQueueType() == dma::DmaQueue::QueueType::TmpToSbuf,
               "DMA block to TPB without weights must use input or tmp-to-sbuf queue");
    }

    m_BlockId = m_DmaDescription->gBlockIdForQueue(m_Queue);
}

/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockTpbToTpb::DmaBlockTpbToTpb(DmaDescription* dmaDescription,
            const dma::DmaQueue* que, EngineId engId, const char* comment)
    : DmaBlockNonIo(dmaDescription, que, engId, comment)
{
    Assert(que->gQueueType() == dma::DmaQueue::QueueType::SbufToSbuf,
           "DMA TPB-to-TPB block must use Sbuf-to-Sbuf queue");
    m_BlockId = m_DmaDescription->gBlockIdForQueue(m_Queue);
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaBlockToTpb::addDmaDesc(TongaAddress srcFileAddress,
        const std::string& refFile,
        TpbAddress dstTpbSbAddress, kcc_int32 numBytes)
{
    DmaDescToTpb desc(numBytes, dstTpbSbAddress, srcFileAddress,
                      m_DmaDescription->gFileSymbolicId(refFile, FileType::Weight));

    m_Descs.push_back(desc);
}


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockFromTpb::DmaBlockFromTpb(DmaDescription* dmaDescription,
        const dma::DmaQueue* que, EngineId engId, bool qOut, const char* comment)
    : DmaBlockNonIo(dmaDescription, que, engId, comment)
    , m_QOut(qOut)
{
    Assert(que->gQueueType() == dma::DmaQueue::QueueType::Output
           || que->gQueueType() == dma::DmaQueue::QueueType::SbufToTmp,
           "DMA block from TPB must use output or sbuf-to-tmp queue");
    m_BlockId = m_DmaDescription->gBlockIdForQueue(m_Queue);
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaBlockFromTpb::addDmaDesc(TpbAddress srcTpbSbAddress,
        TongaAddress dstFileAddress,
        const std::string& refFile,
        kcc_int32 numBytes)
{
    const FileIdType idType = m_DmaDescription->gFileSymbolicId(refFile, FileType::TmpBuffer);
    DmaDescFromTpb desc(numBytes, srcTpbSbAddress, dstFileAddress, idType);

    m_Descs.push_back(desc);
}


/***********************************************************************
***********************************************************************/
DmaDescription::DmaBlockInput::DmaBlockInput(DmaDescription* dmaDescription,
            const dma::DmaQueue* que, EngineId engId, const char* comment)
    : DmaBlock(dmaDescription, que, comment)
    , m_EngineId(engId)
{
    Assert(que->gQueueType() == dma::DmaQueue::QueueType::Input,
           "DMA input block must use input queue");
    m_BlockId = m_DmaDescription->gBlockIdForQueue(m_Queue);
}



/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaBlockInput::addDmaDesc(TongaAddress inputAddress,
        TongaAddress dstSbAddress,
        const std::string& refFile,
        kcc_int32 numBytes)
{
    const FileIdType& idType(m_DmaDescription->gInFileSymbolicId(refFile));
    DmaDescToTpb desc(numBytes, dstSbAddress, inputAddress, idType);

    m_Descs.push_back(desc);
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaBlockTpbToTpb::addDmaDesc(
        kcc_int32 numBytes,
        TpbAddress srcSbAddress,
        TpbAddress dstSbAddress)
{
    DmaDescTpbToTpb desc(numBytes, srcSbAddress,dstSbAddress);
    m_Descs.push_back(desc);
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaDescFromTpb::assertAccessCheck() const
{
    const arch::StateBuffer stateBuf(arch::Arch::gArch().gStateBuffer());
    const kcc_uint32 size    = gNumBytes();
    const TpbAddress sbReadAddr  = gSrcSbAddress();
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
    const TpbAddress sbWriteAddr = gDstSbAddress();
    const kcc_uint32 size   = gNumBytes();
    Assert(stateBuf.qTpbWriteAccessCheck(sbWriteAddr, size),
        "Unaligned DMA state buffer write access. Addr=",
        std::hex, sbWriteAddr, std::dec, " size=", size);
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::DmaDescTpbToTpb::assertAccessCheck() const
{
    const arch::StateBuffer stateBuf(arch::Arch::gArch().gStateBuffer());
    const kcc_uint32 size   = gNumBytes();

    const TpbAddress sbReadAddr = gSrcSbAddress();
    Assert(stateBuf.qTpbReadAccessCheck(sbReadAddr, size),
        "Unaligned DMA state buffer read access. Addr=",
        std::hex, sbReadAddr, std::dec, " size=", size);

    const TpbAddress sbWriteAddr = gDstSbAddress();
    Assert(stateBuf.qTpbWriteAccessCheck(sbWriteAddr, size),
        "Unaligned DMA state buffer write access. Addr=",
        std::hex, sbWriteAddr, std::dec, " size=", size);
}

}}


