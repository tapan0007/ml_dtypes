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
#include "wave/inc/sbatomloadwaveop.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodetpbcopy.hpp"
#include "wavecode/inc/wavecodesbatomload_kelf.hpp"

namespace kcc {
namespace wavecode {




//************************************************************************
WaveCodeSbAtomLoadKelf::WaveCodeSbAtomLoadKelf(WaveCodeRef waveCode)
    : WaveCodeSbAtomLoad(waveCode)
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
WaveCodeSbAtomLoadKelf::generate(wave::WaveOp* waveOp)
{
    Assert(qGenerateKelf(), "Must be in Kelf mode to generate Load for kelf");
    const auto sbAtomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveOp);
    assert(sbAtomLoadWaveop);

    const kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    calcInputSize(sbAtomLoadWaveop);
    if (sbAtomLoadWaveop->qContainWeights()) {
        generateForKelf(sbAtomLoadWaveop);
    } else {
        if (kelfDma.qHasFile(sbAtomLoadWaveop->gRefFileName())) {
            generateForKelf(sbAtomLoadWaveop); // intermediate load
        } else {
            generateInputDma(sbAtomLoadWaveop);
        }
    }
}



//======================================================================
kcc_int32
WaveCodeSbAtomLoadKelf::generateForKelf(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;
    const kcc_int32 numSyncs = findSuccEventsAndChosenEngine(sbAtomLoadWaveop,
                                        chosenEngId, succEventIds);

    if (sbAtomLoadWaveop->gIfmapReplicationResolution() == 0) {
        generateDmaDescAndTriggerRuntimeKelf(sbAtomLoadWaveop, chosenEngId, succEventIds);
    } else {
        generateDmaDescAndTriggerRuntimeKelfWithReplication(sbAtomLoadWaveop, chosenEngId, succEventIds);
    }

    return numSyncs;
}

//======================================================================
void
WaveCodeSbAtomLoadKelf::generateDmaDescAndTriggerRuntimeKelf(
        wave::SbAtomLoadWaveOp* sbAtomLoadWaveop,
        const EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart()
                                        ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    const bool weights              = sbAtomLoadWaveop->qContainWeights();
    Assert(!weights || succEventIds.size() <= 0, "Events not allowed on Load weights");

    //************************************************************************
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());

    std::ostringstream oss;
    oss << sbAtomLoadWaveop->gOrder() << "-" << sbAtomLoadWaveop->gName();

    kelf::DmaDescription::DmaBlockToTpb* dmaBlockToTpb0 = nullptr;
    kelf::DmaDescription::DmaBlockToTpb* dmaBlockToTpb1 = nullptr;
    const kcc_int32 blockIdx0 = kelfDma.startNewDmaBlockToTpb(sbAtomLoadWaveop->gDmaQueue(), chosenEngId, weights, oss.str().c_str());
    const auto que1 = sbAtomLoadWaveop->gDmaQueue1();
    if (que1) {
        Assert(weights, "Cannot have 2 DMA queues for non-weights");
        oss << "-1";
        const kcc_int32 blockIdx1 = kelfDma.startNewDmaBlockToTpb(que1, chosenEngId, weights, oss.str().c_str());
        // both gDmaBlockToTpb() calls must occur AFTER both blocks have been started
        dmaBlockToTpb1 = kelfDma.gDmaBlockToTpb(blockIdx1);
        dmaBlockToTpb0 = kelfDma.gDmaBlockToTpb(blockIdx0);
    } else {
        dmaBlockToTpb0 = kelfDma.gDmaBlockToTpb(blockIdx0);
    }

    const std::string& refFileName(sbAtomLoadWaveop->gRefFileName());

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        const TongaAddress fileAddress = sbAtomLoadWaveop->gOffsetInFile() + (partIdx * stepSize);
        const TpbAddress sbTpbAddress = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        if (!dmaBlockToTpb1 || (partIdx < startPart + numPartitions/2)) {
            dmaBlockToTpb0->addDmaDesc(fileAddress, refFileName, sbTpbAddress, numBytesPerPart);
        } else {
            dmaBlockToTpb1->addDmaDesc(fileAddress, refFileName, sbTpbAddress, numBytesPerPart);
        }
    }

    for (auto eventId : succEventIds) {
        dmaBlockToTpb0->addTailEventId(eventId);
    }
    //************************************************************************
    // Incoming events were processed in
    // WaveCodeSbAtomLoadKelf::findSuccEventsAndChosenEngine(),
    // so processing them again in processIncomingEdgesForceWait is wrong.
    // The event on the chosen pred edge is replaced by sequential execution of
    // the previous waveop and TRIGGER.
    //
    // Of non-chosen pred edges, one of the Waits can be implemented as embedded.
    // The current code does not do that yet.

    if (false && qParallelStreams()) { // incoming events, adds wait for each
        events::EventId waitEventId = 0; // events::EventId_Invalid();
        events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

        processIncomingEdgesForceWait(sbAtomLoadWaveop, chosenEngId, waitEventId, waitEventMode);
    } // end incoming events

    //************************************************************************
    addDmaBarrier(sbAtomLoadWaveop, chosenEngId);


    //************************************************************************
    compisa::DmaTriggerInstr dmaTriggerInstr;
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_idx, 0); // succ evt is in the descriptor block
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    AssignWithSizeCheck(dmaTriggerInstr.use_raw_count, 0); // get from JSON

    AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaBlockToTpb0->gBlockId());
    dmaTriggerInstr.SetDmaQueueName(dmaBlockToTpb0->gDmaQueue()->gName().c_str());

    {
        std::ostringstream oss;
        oss << sbAtomLoadWaveop->gOrder()
            << ":" << (succEventIds.size() > 0 ? succEventIds[0] : -1)
            << "-" << sbAtomLoadWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);


        if (dmaBlockToTpb1) {
            dmaTriggerInstr.SetDmaQueueName(dmaBlockToTpb1->gDmaQueue()->gName().c_str());
            AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaBlockToTpb1->gBlockId());
            oss << "-1";
            m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
            m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
        }
    }
}

//======================================================================
void
WaveCodeSbAtomLoadKelf::generateInputDma(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    //kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());

    if (sbAtomLoadWaveop->gIfmapReplicationResolution() == 0) {
        generateInputDmaNoRepl(sbAtomLoadWaveop);
    } else {
        generateInputDmaRepl(sbAtomLoadWaveop);
    }
}

//======================================================================
void
WaveCodeSbAtomLoadKelf::generateInputDmaRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    if (sbAtomLoadWaveop->gPairCopyWaveOp()) {
        generateInputDmaReplWithCopy(sbAtomLoadWaveop);
    } else {
        generateInputDmaReplWithoutCopy(sbAtomLoadWaveop);
    }
}

//======================================================================
void
WaveCodeSbAtomLoadKelf::generateInputDmaReplWithCopy(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    m_LoadedFileToSbufAddress.clear();

    compisa::DmaTriggerInstr dmaTriggerInstr;
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_idx, 0); // succ evt is in the descriptor block
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;

    std::ostringstream oss;
    oss << sbAtomLoadWaveop->gOrder() << "-" << sbAtomLoadWaveop->gName();
    /*const kcc_int32 numSyncs =*/ findSuccEventsAndChosenEngine(sbAtomLoadWaveop,
                                        chosenEngId, succEventIds);
    Assert(succEventIds.size() <= 0, "Cannot have event IDs on load/save/copy");
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    const kcc_int32 blockIdx = kelfDma.startNewDmaBlockInput(sbAtomLoadWaveop->gDmaQueue(), chosenEngId, oss.str().c_str());
    kelf::DmaDescription::DmaBlockInput* dmaInBlock = kelfDma.gDmaBlockInput(blockIdx);

    for (auto& transferRange : m_WaveCodeTpbCopy->gNotCopied()) {
        const auto& fileRange(std::get<0>(transferRange));
        const auto sbAddress(std::get<1>(transferRange));

        dmaInBlock->addDmaDesc(fileRange.gBegin(), sbAddress, fileRange.gFile(), fileRange.gSize());
        m_LoadedFileToSbufAddress[fileRange] = sbAddress;
        if (REPL_DEBUG) {
            std::cout << "Load " << sbAtomLoadWaveop->gName() << " "
                << " LOADING " << fileRange.String()
                << " to SB " << sbAddress << "\n";
        }
    }

    dmaTriggerInstr.SetDmaQueueName(dmaInBlock->gDmaQueue()->gName().c_str());
    AssignWithSizeCheck(dmaTriggerInstr.use_raw_count, 0);
    AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaInBlock->gBlockId());
    {
        std::ostringstream oss;
        oss << sbAtomLoadWaveop ->gOrder()
            << "-" << sbAtomLoadWaveop ->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }

    addDmaBarrier(sbAtomLoadWaveop, chosenEngId);
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
    const auto pairCopy = sbAtomLoadWaveop->gPairCopyWaveOp();
    if (pairCopy) {
        m_WaveCodeTpbCopy->writeDmaTriggerInstruction();
    }
} // generateInputDmaReplWithCopy(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)

//======================================================================
void
WaveCodeSbAtomLoadKelf::generateInputDmaReplWithoutCopy(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    const utils::DataType& dataType(sbAtomLoadWaveop->gDataType());
    //************************************************************************
    //const kcc_int64 numPartitions   = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    //const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart()
                                        ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    const kcc_int32 replResolution  = sbAtomLoadWaveop->gIfmapReplicationResolution();
    const kcc_int32 replStepElem    = sbAtomLoadWaveop->gSrcStepElem();
    const kcc_int32 numActiveParts  = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int32 ifmapReplNumRows    = sbAtomLoadWaveop->gIfmapReplicationNumRows();
    const kcc_int32 ifmapReplStepBytes  = sbAtomLoadWaveop->gIfmapReplicationStepBytes();

    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;

    /*const kcc_int32 numSyncs =*/ findSuccEventsAndChosenEngine(sbAtomLoadWaveop,
                                        chosenEngId, succEventIds);

    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());

    std::ostringstream oss;
    oss << sbAtomLoadWaveop->gOrder() << "-" << sbAtomLoadWaveop->gName();
    //************************************************************************
    const kcc_int32 blockIdx = kelfDma.startNewDmaBlockInput(sbAtomLoadWaveop->gDmaQueue(), chosenEngId, oss.str().c_str());
    kelf::DmaDescription::DmaBlockInput* dmaInBlock = kelfDma.gDmaBlockInput(blockIdx);
    {
        const TongaAddress sbPartStep = stateBuf.gEntryTpbAddress(1, 0) - stateBuf.gEntryTpbAddress(0, 0);
        const kcc_int32 stride       = replStepElem;
        const kcc_int32 numInChans    = replResolution / stride;
        Assert(numInChans*stride == replResolution,
                "Num in channels (", numInChans, ") * stride (", stride,
                ") != Replication resolution (", replResolution, ")");
        const TongaAddress  partStepBytes       = sbAtomLoadWaveop->gPartitionStepBytes();

        // Compute the offset within FMAP to get to odd elements (for stride=2)
        const kcc_uint32 strideNumBytes = partStepBytes / stride;
        Assert(strideNumBytes * stride == partStepBytes, "Part step bytes not divisible by stride");

        // TODO: use the new ref_file_sz field in waveop
        // assume that H*W*dtypeSize == partStepBytes since when replication there is no input chan folding
        //const std::array<kcc_int32,4>& refFileShape(sbAtomLoadWaveop->gRefFileShape());
        const utils::TensorParams tensorParams(sbAtomLoadWaveop->gRefFileShape(),
                                               sbAtomLoadWaveop->gRefFileFormat().c_str());
        kcc_uint64 inputSize = dataType.gSizeInBytes();
        for (auto n : tensorParams) {
            inputSize *= n;
        }

        const TongaAddress    fileStartAddress  = sbAtomLoadWaveop->gOffsetInFile();
        TpbAddress    sbGroupTpbAddress = stateBuf.gEntryTpbAddress(startPart, addressInPart);

        // Compute the row offset within FMAP to get to odd rows (for stride=2)
        const kcc_uint64 fmapH         = strideNumBytes / ifmapReplStepBytes;
        //const kcc_uint64 strideNumRows = ceil(float(fmapH) / stride);
        const kcc_uint64 strideNumRows = (fmapH + stride-1) / stride;

        m_LoadedFileToSbufAddress.clear();

        // Keep track of group count to determine even/odd row
        // TODO: to be correct for all cases, need to recompute the actual starting row
        // (in this first layer of ResNet, starting row happens to be even for all tiles)
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
                                std::cout << "Trimming SbAtomLoad: "
                                        << sbAtomLoadWaveop->gName() << "\n"
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

                        const FileRange fileRange(sbAtomLoadWaveop->gRefFileName(), filePartAddress, numBytesToWrite);

                        dmaInBlock->addDmaDesc(filePartAddress, sbTpbAddress,
                                   sbAtomLoadWaveop->gRefFileName(), numBytesToWrite);
                        m_LoadedFileToSbufAddress[fileRange] = sbTpbAddress;
                        // ****************************************************************
                        if (REPL_DEBUG) {
                            std::ostringstream oss;
                            oss << "(pp=" << activePartCnt + pp << ",p=" << activePartCnt
                                << ",s=" << strideIdx << ",c=" << c_idx << ")";
                            std::cout << "Load " << sbAtomLoadWaveop->gName() << " "
                                << oss.str() << " LOADING " << fileRange.String()
                                << " to SB " << sbTpbAddress << "\n";
                        }
                        // ****************************************************************


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
    }
    if (dmaInBlock->size() <= 0) {
        // This avoids empty descriptor blocks
        const TpbAddress address = stateBuf.gEntryTpbAddress(127, 0);
        dmaInBlock->addDmaDesc(0, address, sbAtomLoadWaveop->gRefFileName(), 1);
    }

    for (auto eventId : succEventIds) {
        dmaInBlock->addTailEventId(eventId);
    }

    //************************************************************************

    addDmaBarrier(sbAtomLoadWaveop, chosenEngId);
    compisa::DmaTriggerInstr dmaTriggerInstr;
    dmaTriggerInstr.SetDmaQueueName(dmaInBlock->gDmaQueue()->gName().c_str());
    AssignWithSizeCheck(dmaTriggerInstr.use_raw_count, 0); // get from JSON
    AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaInBlock->gBlockId());

    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));

    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_idx, 0); // succ evt is in the descriptor block
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));
    {
        std::ostringstream oss;
        oss << sbAtomLoadWaveop->gOrder()
            << ":" << (succEventIds.size() > 0 ? succEventIds[0] : -1)
            << "-" << sbAtomLoadWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
} // WaveCodeSbAtomLoadKelf::generateInputDmaReplWithoutCopy(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)


//======================================================================
void
WaveCodeSbAtomLoadKelf::generateInputDmaNoRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart()
                                        ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;

    /*const kcc_int32 numSyncs =*/ findSuccEventsAndChosenEngine(sbAtomLoadWaveop,
                                        chosenEngId, succEventIds);
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());

    std::ostringstream oss;
    oss << sbAtomLoadWaveop->gOrder() << "-" << sbAtomLoadWaveop->gName();
    //************************************************************************
    const kcc_int32 blockIdx = kelfDma.startNewDmaBlockInput(sbAtomLoadWaveop->gDmaQueue(), chosenEngId, oss.str().c_str());
    kelf::DmaDescription::DmaBlockInput* dmaInBlock = kelfDma.gDmaBlockInput(blockIdx);

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        const kcc_uint64 fileAddress = sbAtomLoadWaveop->gOffsetInFile() + (partIdx * stepSize);
        const TpbAddress sbAddress = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        dmaInBlock->addDmaDesc(fileAddress, sbAddress, sbAtomLoadWaveop->gRefFileName(), numBytesPerPart);
    }
    for (auto eventId : succEventIds) {
        dmaInBlock->addTailEventId(eventId);
    }

    //************************************************************************
    addDmaBarrier(sbAtomLoadWaveop, chosenEngId);
    compisa::DmaTriggerInstr dmaTriggerInstr;
    dmaTriggerInstr.SetDmaQueueName(dmaInBlock->gDmaQueue()->gName().c_str());
    AssignWithSizeCheck(dmaTriggerInstr.use_raw_count, 0); // get from JSON
    AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaInBlock->gBlockId());

    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));

    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_idx, 0); // succ evt in the descr. block
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));
    {
        std::ostringstream oss;
        oss << sbAtomLoadWaveop->gOrder()
            << ":" << (succEventIds.size() > 0 ? succEventIds[0] : -1)
            << "-" << sbAtomLoadWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
} // WaveCodeSbAtomLoadKelf::generateInputDmaNoRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)

//======================================================================
// If ifmap_replication_resolution = 2 * 3 (stride * C, where C is # channels), and contains_weights=False, and src_step_elem=2 (stride)
// (only applicable when contains_weights=False), perform:
// 1- sim_dma_copy 3 channels (C) starting at offset_in_file to sb_address (src_step_elem=2 indicates even indices)
// 2- sim_dma_copy 3 channels (C) starting at offset_in_file + 2 bytes (2 bytes for 1 float16 element), to sb_address for channels following #1 (odd indices)
// 3a- increment sb_address by ifmap_replication_num_rows * SB partition size (ifmap_replication_num_rows is 21 for ResNet50 case)
// 3b- increment offset_in_file by ifmap_replication_step_bytes (W *stride)
// 4- if number of channels copied so far < ifmap_count, goto #1
// (we can generalize this better; in particular, stop as soon as number of channels copied so far >= ifmap_count which is number of active rows)
// (2bytes indicates data item size for float16)
//======================================================================
//
// If ifmap_replication_resolution= 3 (C, maybe assert), and contains_weights=True (src_step_elem should be 1 here, but not used, weights do not have stride), perform:
// 1- sim_dma_copy ifmap_replication_resolution channels (C) starting at offset_in_file to sb_address
// 2a- increment sb_address by ifmap_replication_resolution * SB partition size (ifmaps_replication_num_rows is 3 also, different from above, so can also
//     use it instead of ifmap_replication_resolution)
// 2b- increment offset_in_file by ifmap_replication_step_bytes (equals M, maybe assert)
// 3-  if number of channels copied so far < ifmap_count, goto #1
//
// The test for this is 3-1conv0_padvalid_wave.
//======================================================================
// In the previous comment, please ignore ifmap_replication_step_bytes for MatMul waveop.
// Also, for weights SBAtomLoad waveop, ifmap_replication_resolution is different from that in IFMAP SBAtomLoad waveop (3 instead of 6 for ResNet50 first layer).
// Also, step #1 should say:
// 1- sim_dma_copy ifmap_replication_resolution channels starting at offset_in_file to sb_address
//======================================================================


void
WaveCodeSbAtomLoadKelf::generateDmaDescAndTriggerRuntimeKelfWithReplication(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop,
                    const EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    //const utils::DataType& dataType(sbAtomLoadWaveop->gDataType());
    //const kcc_int32 dtypeSize = dataType.gSizeInBytes();

   //************************************************************************
    const kcc_int64 addressInPart       = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 startPart           = sbAtomLoadWaveop->gStartAtMidPart()
                                            ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    const bool      qWeights            = sbAtomLoadWaveop->qContainWeights();

    const kcc_int32 replResolution      = sbAtomLoadWaveop->gIfmapReplicationResolution();
    const kcc_int32 ifmapReplStepBytes  = sbAtomLoadWaveop->gIfmapReplicationStepBytes();
    const kcc_int32 replStepElem        = sbAtomLoadWaveop->gSrcStepElem();
    const kcc_int32 numActiveParts      = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int32 ifmapReplNumRows    = sbAtomLoadWaveop->gIfmapReplicationNumRows();
    const kcc_int64 numBytesPerPart     = sbAtomLoadWaveop->gLength();

    const std::string& refFileName(sbAtomLoadWaveop->gRefFileName());
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    kelf::DmaDescription::DmaBlockToTpb* dmaBlockToTpb0 = nullptr;
    kelf::DmaDescription::DmaBlockToTpb* dmaBlockToTpb1 = nullptr;

    std::ostringstream oss;
    oss << sbAtomLoadWaveop->gOrder() << "-" << sbAtomLoadWaveop->gName();
    if (qWeights) {
        Assert(succEventIds.size() <= 0, "Events not allowed on Load weights");
        Assert(1 == replStepElem, "Load weights should have stride 1 in replication");
        Assert(numBytesPerPart == ifmapReplStepBytes, "Replication step bytes should be equal to length");
        const kcc_uint32 numChans    = replResolution;
        const TpbAddress  partStepBytes       = sbAtomLoadWaveop->gPartitionStepBytes();
        const TpbAddress  sbPartStep = stateBuf.gEntryTpbAddress(1, 0) - stateBuf.gEntryTpbAddress(0, 0);
        Assert(1 == replStepElem, "Load weights should have stride 1 in replication");
        Assert(numBytesPerPart == ifmapReplStepBytes, "Replication step bytes should be equal to length");


        const kcc_int32 blockIdx0 = kelfDma.startNewDmaBlockToTpb(sbAtomLoadWaveop->gDmaQueue(), chosenEngId, qWeights, oss.str().c_str());
        const auto que1 = sbAtomLoadWaveop->gDmaQueue1();
        if (que1) {
            oss << "-1";
            const kcc_int32 blockIdx1 = kelfDma.startNewDmaBlockToTpb(que1, chosenEngId, qWeights, oss.str().c_str());
            // both gDmaBlockToTpb() calls must occur AFTER both blocks have been started
            dmaBlockToTpb1 = kelfDma.gDmaBlockToTpb(blockIdx1);
            dmaBlockToTpb0 = kelfDma.gDmaBlockToTpb(blockIdx0);
        } else {
            dmaBlockToTpb0 = kelfDma.gDmaBlockToTpb(blockIdx0);
        }


        TongaAddress fileAddress     = sbAtomLoadWaveop->gOffsetInFile();
        TpbAddress   sbTpbAddress    = stateBuf.gEntryTpbAddress(startPart, addressInPart);


        kcc_int32 part = startPart;
        while (part < startPart + numActiveParts) {
            for (kcc_uint32 c_idx = 0; c_idx < numChans; ++c_idx) {
                const TongaAddress currFileAddress = fileAddress + c_idx * partStepBytes;
                if (!dmaBlockToTpb1 || part < startPart + numActiveParts/2) {
                    dmaBlockToTpb0->addDmaDesc(currFileAddress, refFileName, sbTpbAddress, ifmapReplStepBytes);
                } else {
                    dmaBlockToTpb1->addDmaDesc(currFileAddress, refFileName, sbTpbAddress, ifmapReplStepBytes);
                }

                sbTpbAddress += sbPartStep;   // += 128k
                ++part;
            }
            fileAddress += ifmapReplStepBytes;  // += 128
        }

    } else {
        const kcc_int32 blockIdx0 = kelfDma.startNewDmaBlockToTpb(sbAtomLoadWaveop->gDmaQueue(), chosenEngId, qWeights, oss.str().c_str());
        kelf::DmaDescription::DmaBlockToTpb* dmaBlockToTpb0 = kelfDma.gDmaBlockToTpb(blockIdx0);

        const TpbAddress    sbPartStep = stateBuf.gEntryTpbAddress(1, addressInPart) - stateBuf.gEntryTpbAddress(0, addressInPart);
        const kcc_int32 numInChans = replResolution / replStepElem;
        Assert(numInChans*replStepElem == replResolution,
                "Num in channels (", numInChans, ") * stride (", replStepElem,
                ") != Replication resolution (", replResolution, ")");

        const kcc_int32     numBytesPerPart     = sbAtomLoadWaveop->gLength();
        const TongaAddress  partStepBytes       = sbAtomLoadWaveop->gPartitionStepBytes();
        TongaAddress        fileGroupAddress    = sbAtomLoadWaveop->gOffsetInFile();
        TpbAddress          sbGroupTpbAddress   = stateBuf.gEntryTpbAddress(startPart, addressInPart);

        // assume that H*W*dtypeSize == partStepBytes since when replication there is no input chan folding
        kcc_int32 activePartCnt = 0;
        while (activePartCnt < numActiveParts) {
            TpbAddress sbTpbAddress = sbGroupTpbAddress;
            TongaAddress fileAddress = fileGroupAddress;
            for (kcc_int32 strideIdx = 0; strideIdx < replStepElem; ++strideIdx) {
                const TongaAddress strideFileOffset = (partStepBytes/replStepElem) * strideIdx;
                for (kcc_int32 chanIdx = 0; chanIdx < numInChans; ++chanIdx) {
                    TongaAddress chanOffset = chanIdx * partStepBytes;
                    TongaAddress filePartAddress = fileAddress + chanOffset + strideFileOffset;
                    dmaBlockToTpb0->addDmaDesc(filePartAddress, refFileName, sbTpbAddress, numBytesPerPart);
                    sbTpbAddress += sbPartStep;
                }
            }
            activePartCnt       += ifmapReplNumRows;
            fileGroupAddress    += ifmapReplStepBytes;
            sbGroupTpbAddress   += ifmapReplNumRows * sbPartStep;
        }
    }

    for (auto eventId : succEventIds) {
        dmaBlockToTpb0->addTailEventId(eventId);
    }


    //************************************************************************
    addDmaBarrier(sbAtomLoadWaveop, chosenEngId);
    //************************************************************************
    {
        compisa::DmaTriggerInstr dmaTriggerInstr;

        AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_idx, 0);
        AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
        AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_idx, 0); // succ evt is in the descriptor block
        AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

        AssignWithSizeCheck(dmaTriggerInstr.use_raw_count, 0); // get from JSON


        std::ostringstream oss;
        oss << sbAtomLoadWaveop->gOrder()
            << ":" << (succEventIds.size() > 0 ? succEventIds[0] : -1)
            << "-" << sbAtomLoadWaveop->gName();

        dmaTriggerInstr.SetDmaQueueName(dmaBlockToTpb0->gDmaQueue()->gName().c_str());
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
        AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaBlockToTpb0->gBlockId());
        m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);

        if (dmaBlockToTpb1) {
            oss << "-1";
            dmaTriggerInstr.SetDmaQueueName(dmaBlockToTpb1->gDmaQueue()->gName().c_str());
            m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
            AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaBlockToTpb1->gBlockId());
            m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
        }

    }
} // WaveCodeSbAtomLoadKelf::generateDmaDescAndTriggerRuntimeKelfWithReplication



/***********************************************************************
***********************************************************************/
bool
WaveCodeSbAtomLoadKelf::Loaded(const FileRange& tongaRange, TpbAddress& srcAddress,
    OffsetRange& loadedRangeRet) const
{
    const auto& requestedRange = tongaRange.gOffsetRange();

    auto it = m_LoadedFileToSbufAddress.find(tongaRange);
    if (it != m_LoadedFileToSbufAddress.end()) {
        // exact match
        srcAddress = (*it).second;
        loadedRangeRet = requestedRange;
        return true;
    }
    for (auto it : m_LoadedFileToSbufAddress) {
        if (it.first.gFile() != tongaRange.gFile()) {
            continue;
        }
        const auto& existingRange(it.first.gOffsetRange());
        // existingRange.Begin is loaded to (it).second
        if (requestedRange.gBegin() <= existingRange.gBegin()
            &&                         existingRange.gBegin() < requestedRange.gEnd()
            &&                                                  requestedRange.gEnd() <= existingRange.gEnd())
        {
            //            v
            // case A: req.B  <=  exist.B   <   req.E <= exist.E => overlap is [exist.b, req.E)
            //            ^            ^           ^
            //            ^B.unloaded.E^-B.loaded.E^
            loadedRangeRet = OffsetRange(existingRange.gBegin(), requestedRange.gEnd() - existingRange.gBegin());

            srcAddress = (it).second;
            return true;
        }
        if (existingRange.gBegin() <= requestedRange.gBegin()
            &&                        requestedRange.gBegin() < existingRange.gEnd()
            &&                                                  existingRange.gEnd() <= requestedRange.gEnd())
        {
            // case B: exist.B <= req.B < exist.E  <=  req.E -> [req.b, exist.E)
            //                      ^--loaded--^-unloaded-^
            loadedRangeRet = OffsetRange(requestedRange.gBegin(), existingRange.gEnd() - requestedRange.gBegin());

            srcAddress = (it).second + (requestedRange.gBegin() - existingRange.gBegin());
            return true;
        }
    }

    return false;
}

}}

