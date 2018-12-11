#include <sstream>

#include "utils/inc/debug.hpp"
#include "utils/inc/misc.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "compisa/inc/compisadmatrigger.hpp"


#include "events/inc/events.hpp"



#include "wave/inc/waveedge.hpp"
#include "wave/inc/tpbcopywaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomload_kelf.hpp"
#include "wavecode/inc/wavecodetpbcopy.hpp"

namespace kcc {
namespace wavecode {




//************************************************************************
WaveCodeTpbCopy::WaveCodeTpbCopy(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}




//************************************************************************
// Suppose an SbLoad has the following predecessors and successors
// MM1          POOL        |
//   \          /           |
//    \        /            |
//   e1\      /e2           |
//      \    /              |
//       \  /               |
//       SbLoad             |
//       /  \               |
//      /    \              |
//   e3/      \e4           |
//    /        \            |
//  MM2        ACT          |
//
//************************************************************************
// With the above wavegraph, on SIM the instructions would be
//
//  PeEng    PoolEng   ActEng   AngelEng        |
//  ...       ...      ...      ...             |
//  MM1      POOL               ...             |
//  Set e1   Set e2                             |
//           ...                Wait e1         |
//                              Wait e2         |
//           ...                MemCpy          |
//                              Set e3          |
//                              Set e4          |
//  Wait e3  ...      Wait e4   ...             |
//  MM2      ...      ACT       ...             |
//
//************************************************************************
// To execute DMA on one of the calculation engines:
// 1. Pick one of the incoming engines to execute DmaTrigger
// 2. Pick one of the outgoing engines to wait for Dma
//
//************************************************************************
// Suppose we picked PoolEng to start DMA
// PoolEng will wait for all events that Sb
//
//  PeEng         PoolEng        ActEng          |
//  ...            ...           ...            |
//  MM1           POOL                          |
//  Set e1                                      |
//                Wait e1                       |
//                DmaStart(e3,e4)               |
//                 ...                          |
//                               Wait e4        |
//  Wait e3        ...           ACT            |
//  MM2            ...           ...            |
//
// Code for PeEng and ActEng is unchanged
// All waits by SbLoad are transferred to Pool except for
// the wait Pool->SbLoad.
// Pool executes DmaStart with all events that DMA sets.
//
//
//************************************************************************

//************************************************************************
void
WaveCodeTpbCopy::generate(wave::WaveOp* waveOp)
{
    Assert(qGenerateKelf(), "Must be in Kelf mode to generate TpbCopy for kelf");
    const auto tpbCopyWaveop = dynamic_cast<wave::TpbCopyWaveOp*>(waveOp);
    Assert(tpbCopyWaveop, "Must be TpbCopy");
    const auto pairLoad = tpbCopyWaveop->gPairLoadWaveOp();
    Assert(pairLoad, "TpbCopy waveop '", tpbCopyWaveop->gName(), "' has no Load pair");

    //m_CopiedFileToSbufAddress.clear();
    m_NotCopiedFileToSbufAddress.clear();

    const utils::DataType& dataType(pairLoad->gDataType());
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const kcc_int64 numBytesPerPart     = pairLoad->gLength();
    const kcc_int32 stride              = pairLoad->gSrcStepElem();
    const kcc_int64 startPart           = pairLoad->gStartAtMidPart();
    const kcc_int32 numActiveParts      = pairLoad->gNumPartitions();
    const kcc_int32 ifmapReplNumRows    = pairLoad->gIfmapReplicationNumRows();
    const kcc_int64 addressInPart       = pairLoad->gSbAddress();
    const TongaAddress  partStepBytes   = pairLoad->gPartitionStepBytes();
    const kcc_int32 ifmapReplStepBytes  = pairLoad->gIfmapReplicationStepBytes();
    const kcc_int32 replResolution      = pairLoad->gIfmapReplicationResolution();
    const TongaAddress fileStartAddress = pairLoad->gOffsetInFile();

    const kcc_uint32 strideNumBytes     = partStepBytes / stride;
    const kcc_uint64 fmapH              = strideNumBytes / ifmapReplStepBytes;
    //const kcc_uint64 strideNumRows      = ceil(float(fmapH) / stride);
    const kcc_uint64 strideNumRows      = (fmapH + stride-1) / stride;
    Assert(strideNumBytes * stride == partStepBytes, "Part step bytes not divisible by stride");
    const kcc_int32 numInChans          = replResolution / stride;
    Assert(numInChans*stride == replResolution,
                "Num in channels (", numInChans, ") * stride (", stride,
                ") != Replication resolution (", replResolution, ")");
    const TongaAddress sbPartStep = stateBuf.gEntryTpbAddress(1, 0) - stateBuf.gEntryTpbAddress(0, 0);

    kcc_uint64 inputSize = dataType.gSizeInBytes();
    const utils::TensorParams tensorParams(pairLoad->gRefFileShape(),
                                           pairLoad->gRefFileFormat().c_str());
    for (auto n : tensorParams) {
        inputSize *= n;
    }
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    std::ostringstream oss;
    oss << tpbCopyWaveop->gOrder() << "-" << tpbCopyWaveop->gName();

    EngineId chosenEngId = tpbCopyWaveop->gEngineId();
    //************************************************************************
    compisa::DmaTriggerInstr dmaTriggerInstr;
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_idx, 0); // succ evt is in the descriptor block
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    processIncomingEdges(tpbCopyWaveop, chosenEngId);

    //************************************************************************
    const kcc_int32 blockIdx = kelfDma.startNewDmaBlockTpbToTpb(tpbCopyWaveop->gDmaQueue(), chosenEngId, oss.str().c_str());
    auto dmaTpbToTpbBlock = kelfDma.gDmaBlockTpbToTpb(blockIdx);
    //************************************************************************
    TpbAddress    sbGroupTpbAddress = stateBuf.gEntryTpbAddress(startPart, addressInPart);

    kcc_int32 fileGroupCnt = 0;
    kcc_int32 activePartCnt = startPart;

    while (activePartCnt < startPart + numActiveParts) {
        TpbAddress sbTpbAddress = sbGroupTpbAddress;
        const TongaAddress fileGroupOffset = ((fileGroupCnt % stride) * strideNumRows + (fileGroupCnt / stride)) * ifmapReplStepBytes;
        const TongaAddress fileGroupAddress = fileStartAddress + fileGroupOffset;

        kcc_int32 pp = 0;
        for (kcc_int32 strideIdx = 0; strideIdx < stride; ++strideIdx) {
            const kcc_uint64 strideOffset = strideIdx * strideNumBytes;  // {0,1}*(105340/2)
            for (kcc_int32 c_idx = 0; c_idx < numInChans; ++c_idx) {
                const TongaAddress chanOffset = c_idx * partStepBytes;  // {0,1,2}*105340
                const TongaAddress filePartAddress = fileGroupAddress + chanOffset + strideOffset;
                if (filePartAddress < inputSize) {
                    kcc_int64 numBytesToWrite;
                    if (filePartAddress + numBytesPerPart <= inputSize) {
                        numBytesToWrite = numBytesPerPart;
                    } else {
                        numBytesToWrite = inputSize - filePartAddress;
                        {
                            std::cout << "Trimming SbAtomLoad:\n"
                                    << "    input size: " << inputSize << "\n"
                                    << "    file address: " << filePartAddress << "\n"
                                    << "    num bytes per part (requested): " << numBytesPerPart << "\n"
                                    << "    num bytes to write: " << numBytesToWrite << "\n"
                                    << "    part: " << activePartCnt << "\n"
                                    << "    stride idx: " << strideIdx << "\n"
                                    << "    chan idx: " << c_idx << "\n"
                                    ;
                            std::cout << "    Fmap tensor: ";
                            for (auto n : tensorParams) {
                                std::cout << n << " ";
                            }
                            std::cout << "\n";
                        }
                    }

                    const FileRange requestedRange(pairLoad->gRefFileName(), filePartAddress, numBytesToWrite);

                    std::ostringstream oss;
                    oss << "(pp=" << activePartCnt + pp << ",p=" << activePartCnt
                        << ",s=" << strideIdx << ",c=" << c_idx << ")";
                    TpbAddress loadedSrcAddress;
                    OffsetRange loadedRange;

                    if (m_WaveCodeSbAtomLoadKelf->Loaded(requestedRange, loadedSrcAddress, loadedRange)) {

                        if (loadedRange.gBegin() == requestedRange.gBegin() && loadedRange.gEnd() == requestedRange.gEnd()) {
                            const FileRange loadedFileRange(requestedRange.gFile(), loadedRange);
                            //m_CopiedFileToSbufAddress[loadedFileRange] = sbTpbAddress;
                            dmaTpbToTpbBlock->addDmaDesc(loadedRange.gEnd()-loadedRange.gBegin(), loadedSrcAddress, sbTpbAddress);
                            if (REPL_DEBUG) {
                                std::cout << "TpbCopy " << tpbCopyWaveop->gName() << " " << oss.str()
                                    << " COPYING whole file range " << loadedFileRange.String()
                                    << " from SB " << loadedSrcAddress << " to SB " << sbTpbAddress << "\n";
                            }
                        } else {
                            Assert(loadedRange.gEnd() == requestedRange.gEnd() || loadedRange.gBegin() == requestedRange.gBegin(),
                                   "loaded range must begin or end of requested range");
                            const FileRange loadedFileRange(requestedRange.gFile(), loadedRange.gBegin(), loadedRange.gSize());

                            TpbAddress   copyTargetSbAddress;
                            TongaAddress laterBegin;
                            TongaAddress laterEnd;
                            TpbAddress   laterLoadTargetSbAddress;

                            const bool endLoaded = loadedRange.gEnd() == requestedRange.gEnd(); // true <==> case A
                            if (endLoaded) {
                                //            v-orig-sb-target corresponds to
                                //            v
                                // case A: req.B <=  exist.B   <   req.E <= exist.E => [exist.b, req.E)
                                //            ^           ^           ^
                                //            ^-unloaded-E^B-LOADED--E^
                                copyTargetSbAddress      = sbTpbAddress + (loadedRange.gBegin() - requestedRange.gBegin());
                                laterBegin               = requestedRange.gBegin();
                                laterEnd                 = loadedRange.gBegin();
                                laterLoadTargetSbAddress = sbTpbAddress;
                            } else {
                                //                       v-orig-sb-target corresponds to
                                //                       v
                                // case B: exist.B <= req.B < exist.E  <=  req.E -> [req.b, exist.E)
                                //                       ^         ^          ^
                                //                       ^--LOADED-^-unloaded-^
                                copyTargetSbAddress      = sbTpbAddress;
                                laterBegin               = loadedRange.gEnd();
                                laterEnd                 = requestedRange.gEnd();
                                laterLoadTargetSbAddress = sbTpbAddress + (loadedRange.gEnd() - loadedRange.gBegin());
                            }
                            dmaTpbToTpbBlock->addDmaDesc(loadedRange.gEnd()-loadedRange.gBegin(), loadedSrcAddress, copyTargetSbAddress);

                            const FileRange nonloadedFileRange(requestedRange.gFile(), laterBegin, laterEnd - laterBegin);
                            const TransferRange transfer(std::make_tuple(nonloadedFileRange, laterLoadTargetSbAddress));
                            Assert(m_NotCopiedFileToSbufAddress.find(transfer) == m_NotCopiedFileToSbufAddress.end(),
                                   "TpbCopy: transfer already saved for loading: from ", nonloadedFileRange.String(), " to ", laterLoadTargetSbAddress);

                            m_NotCopiedFileToSbufAddress.insert(transfer);

                            // ****************************************************************
                            if (REPL_DEBUG) {
                                std::cout << "TpbCopy " << tpbCopyWaveop->gName() << " " << oss.str()
                                    << " COPYING file subrange " << loadedFileRange.String()
                                    << " from SB " << loadedSrcAddress << " to SB " << copyTargetSbAddress << "\n";
                                std::cout << "TpbCopy " << tpbCopyWaveop->gName() << " " << oss.str()
                                    << " DID NOT COPY file subrange " << nonloadedFileRange.String()
                                    << " to SB " << laterLoadTargetSbAddress << "\n";
                            }
                        }
                        // ****************************************************************
                    } else {
                        const TransferRange transfer(std::make_tuple(requestedRange, sbTpbAddress));
                        Assert(m_NotCopiedFileToSbufAddress.find(transfer) == m_NotCopiedFileToSbufAddress.end(),
                               "TpbCopy: range already saved for loading: from ", requestedRange.String(), " to ", sbTpbAddress);
                        m_NotCopiedFileToSbufAddress.insert(transfer);
                        // ****************************************************************
                        if (REPL_DEBUG) {
                            std::cout << "TpbCopy " << tpbCopyWaveop->gName() << " " << oss.str()
                                << " DID NOT COPY whole range " << requestedRange.String() << " to SB "
                                << sbTpbAddress << "\n";
                        }
                        // ****************************************************************
                    }

                }

                sbTpbAddress += sbPartStep;
                ++pp;
            } // for (kcc_int32 c_idx = 0; c_idx < numInChans; ++c_idx)
        } // for (kcc_int32 strideIdx = 0; strideIdx < stride; ++strideIdx)
        if (REPL_DEBUG) {
            std::cout << "\n";
        }

        //fileGroupAddress    += ifmapReplStepBytes;              // += 230  (in the 1st layer of RN50)
        fileGroupCnt++;
        sbGroupTpbAddress += ifmapReplNumRows * sbPartStep;   // += 21*128k  (in the 1st layer of RN50)
        activePartCnt     += ifmapReplNumRows;                // += 21  (in the 1st layer of RN50)
    } // while (activePartCnt < startPart + numActiveParts)

    if (dmaTpbToTpbBlock->size() <= 0) {
        // This avoids empty descriptor blocks
        const TpbAddress address = stateBuf.gEntryTpbAddress(127, 0);
        dmaTpbToTpbBlock->addDmaDesc(1, address, address);
    }

    addDmaBarrier(pairLoad, chosenEngId);
    dmaTriggerInstr.SetDmaQueueName(dmaTpbToTpbBlock->gDmaQueue()->gName().c_str());
    AssignWithSizeCheck(dmaTriggerInstr.use_raw_count, 0); // get from JSON
    AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaTpbToTpbBlock->gBlockId());

    {
        std::ostringstream oss;
        oss << tpbCopyWaveop ->gOrder()
            << "-" << tpbCopyWaveop ->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
}





/***********************************************************************
***********************************************************************/

}}


