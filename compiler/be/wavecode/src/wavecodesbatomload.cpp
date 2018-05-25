#include "utils/inc/debug.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "compisa/inc/compisasimwrnpy.hpp"
#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisadmatrigger.hpp"
#include "compisa/inc/compisasimdmacopy.hpp"


#include "events/inc/events.hpp"


#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/convlayer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomload.hpp"

namespace kcc {
namespace wavecode {




//************************************************************************
WaveCodeSbAtomLoad::WaveCodeSbAtomLoad(WaveCodeRef waveCode)
    : WaveCodeSbAtom(waveCode)
{}

//************************************************************************
void
WaveCodeSbAtomLoad::generate(wave::WaveOp* waveOp)
{
    const auto sbAtomLoadWaveOp = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveOp);
    assert(sbAtomLoadWaveOp);
    if (qGenerateKelf()) {
        if (sbAtomLoadWaveOp->qContainWeights() || m_WaveCode.qBinFileSimKelf()) {
            generateForKelf(sbAtomLoadWaveOp);
        } else {
            generateInputDma(sbAtomLoadWaveOp);
        }
    } else {
        generateForSim(sbAtomLoadWaveOp);
    }
}


//************************************************************************
// Suppose predecessors are w0, w1, w2
// Successors are w3, w4, w5
// We want to issue the following instructions:
// WAIT(w1)
// WAIT(w2)
// MEMCPY first partition with embedded WAIT(w0) and with no-set
// MEMCPY middle partitions with no-wait and no-set
// MEMCPY last partition with no-wait and SET(w3)
// SET(w4)
// SET(w5)
//************************************************************************
void
WaveCodeSbAtomLoad::generateForSim(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp)
{
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomLoadWaveOp->gEngineId();
    Assert(EngineId::DmaEng == engineId, "Engine id for SbAtomLoad waveop should be DmaEng, but is ",
           static_cast<long>(engineId));

    //************************************************************************
    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(sbAtomLoadWaveOp->gRefFileName());
    if (npyFileDramOffset < 0) { // Load whole numpy file to DRAM
        compisa::SimWrNpyInstr npyToDramInstr;
        npyToDramInstr.inst_events.wait_event_idx   = 0;
        npyToDramInstr.inst_events.wait_event_mode  = eventWaitMode2Isa(events::EventWaitMode::DontWait);
        npyToDramInstr.inst_events.set_event_idx    = 0;
        npyToDramInstr.inst_events.set_event_mode   = eventSetMode2Isa(events::EventSetMode::DontSet);

        const kcc_int64 numPySize = sbAtomLoadWaveOp->gLoadDataSizeInBytes();
        strcpy(npyToDramInstr.src_fname, sbAtomLoadWaveOp->gRefFileName().c_str());
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        npyToDramInstr.dst_addr     = npyFileDramOffset;
        m_WaveCode.writeInstruction(npyToDramInstr);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomLoadWaveOp->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomLoadWaveOp->gRefFileShape();
        m_WaveCode.recordDramForNpyFile(sbAtomLoadWaveOp->gRefFileName(), npyFileInfo);
    }

    //************************************************************************
    compisa::SimMemCpyInstr dramToStateBufInstr;
    dramToStateBufInstr.inst_events.wait_event_idx  = 0;
    dramToStateBufInstr.inst_events.wait_event_mode = eventWaitMode2Isa(events::EventWaitMode::DontWait);
    dramToStateBufInstr.inst_events.set_event_idx   = 0;
    dramToStateBufInstr.inst_events.set_event_mode  = eventSetMode2Isa(events::EventSetMode::DontSet);

    events::EventId setEventId = 0; // events::EventId_Invalid();
    events::EventSetMode setEventMode = events::EventSetMode::DontSet;
    events::EventId waitEventId = 0; // events::EventId_Invalid();
    events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(sbAtomLoadWaveOp, waitEventId, waitEventMode);
    } // end incoming events


    if (qParallelStreams()) { // Find first successor for embedded
        findFirstSetEventIdMode(sbAtomLoadWaveOp, setEventId,  setEventMode);
    }

    //************************************************************************
    // Instruction(s)
    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomLoadWaveOp->gIfmapCount();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveOp->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveOp->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveOp->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveOp->gStartAtMidPart() ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;

    dramToStateBufInstr.nbytes      = numBytesPerPart;

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        if (qParallelStreams()) {
            dramToStateBufInstr.inst_events.wait_event_idx      = 0;
            dramToStateBufInstr.inst_events.wait_event_mode     = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
            dramToStateBufInstr.inst_events.set_event_idx       = 0;
            dramToStateBufInstr.inst_events.set_event_mode      = events::eventSetMode2Isa(events::EventSetMode::DontSet);

            if (0 == partIdx) { // only the first reading waits for predecessors
                dramToStateBufInstr.inst_events.wait_event_idx  = waitEventId;
                dramToStateBufInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(waitEventMode);
            }

            if (numPartitions-1 == partIdx) { // only the last reading informs successors
                dramToStateBufInstr.inst_events.set_event_idx   = setEventId;
                dramToStateBufInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(setEventMode);
            }
        }

        dramToStateBufInstr.src_addr = npyFileDramOffset + sbAtomLoadWaveOp->gOffsetInFile() + (partIdx * stepSize);
        dramToStateBufInstr.dst_addr = stateBuf.gEntryTongaAddress(partIdx, addressInPart);

        {
            char buf[256];
            sprintf(buf, "%s-%d", sbAtomLoadWaveOp->gName().c_str(), partIdx);
            SaveName(dramToStateBufInstr, buf);
        }
        m_WaveCode.writeInstruction(dramToStateBufInstr);
    }

    //************************************************************************
    if (qParallelStreams()) { // Write remaining SETs
        processOutgoingEdgesAlreadyEmb(sbAtomLoadWaveOp, setEventId);
    }
}

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

//======================================================================
kcc_int32
WaveCodeSbAtomLoad::generateForKelf(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;
    const kcc_int32 numSyncs = findSuccEventsAndChosenEngine(sbAtomLoadWaveop,
                                        chosenEngId, succEventIds);

    if (m_WaveCode.qBinFileSimKelf()) {
        generateDmaCopySimKelf(sbAtomLoadWaveop, chosenEngId, succEventIds);
    } else {
        generateDmaDescAndTriggerRuntimeKelf(sbAtomLoadWaveop, chosenEngId, succEventIds);
    }

    return numSyncs;
}

//======================================================================
void
WaveCodeSbAtomLoad::generateDmaDescAndTriggerRuntimeKelf(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomLoadWaveop->gIfmapCount();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart()
                                        ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    const bool weights              = sbAtomLoadWaveop->qContainWeights();

    Assert(succEventIds.size() == 1, "AtomLoad: only one succ event id");
    //************************************************************************
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    kelf::DmaDescription::DmaBlockToTpb& dmaBlock(
                        kelfDma.startNewDmaBlockToTpb(chosenEngId, weights));
    const std::string& refFileName(sbAtomLoadWaveop->gRefFileName());

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        const kcc_int64 fileAddress = sbAtomLoadWaveop->gOffsetInFile() + (partIdx * stepSize);
        const TpbAddress sbTpbAddress = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        dmaBlock.addDmaDesc(fileAddress, refFileName, sbTpbAddress, numBytesPerPart);
    }
    for (auto eventId : succEventIds) {
        dmaBlock.addTailEventId(eventId);
    }
    //************************************************************************
    // Incoming events were processed in 
    // WaveCodeSbAtomSave::findSuccEventsAndChosenEngine(),
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
    compisa::DmaTriggerInstr dmaTriggerInstr;
    strncpy(dmaTriggerInstr.dma_queue_name, 
            dmaBlock.gSymbolicQueueName(chosenEngId).c_str(),
            sizeof(dmaTriggerInstr.dma_queue_name)/sizeof(dmaTriggerInstr.dma_queue_name[0]) - 1);
    dmaTriggerInstr.use_raw_count = 0; // get from JSON
    dmaTriggerInstr.block_id = dmaBlock.gBlockId();

    dmaTriggerInstr.inst_events.wait_event_idx  = 0;
    dmaTriggerInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    dmaTriggerInstr.inst_events.set_event_idx   = 0; // succ evt is in the descriptor block
    dmaTriggerInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
    // dummy
    dmaTriggerInstr.inst_events.wait_event_idx  = 0;
    dmaTriggerInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    dmaTriggerInstr.inst_events.set_event_idx   = 0;
    dmaTriggerInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
}

//======================================================================
void
WaveCodeSbAtomLoad::generateInputDma(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomLoadWaveop->gIfmapCount();
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

    kelf::DmaDescription::DmaBlockInput& dmaBlock(kelfDma.startNewDmaBlockInput());

    dmaBlock.addTailEventId(succEventIds[0]);
    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        const kcc_uint64 fileAddress = sbAtomLoadWaveop->gOffsetInFile() + (partIdx * stepSize);
        const TpbAddress sbAddress = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        dmaBlock.addDmaDesc(fileAddress, sbAddress, numBytesPerPart);
    }
}





//======================================================================
void
WaveCodeSbAtomLoad::generateDmaCopySimKelf(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(m_WaveCode.qBinFileSimKelf(), "Must be binary for SIM Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    const kcc_int64 numPartitions   = sbAtomLoadWaveop->gIfmapCount();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart() ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;

    //************************************************************************

    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(sbAtomLoadWaveop->gRefFileName());
    if (npyFileDramOffset < 0) { // Load whole numpy file to DRAM
        compisa::SimWrNpyInstr npyToDramInstr;
        npyToDramInstr.inst_events.wait_event_idx   = 0;
        npyToDramInstr.inst_events.wait_event_mode  = eventWaitMode2Isa(events::EventWaitMode::DontWait);
        npyToDramInstr.inst_events.set_event_idx    = 0;
        npyToDramInstr.inst_events.set_event_mode   = eventSetMode2Isa(events::EventSetMode::DontSet);

        const kcc_int64 numPySize = sbAtomLoadWaveop->gLoadDataSizeInBytes();
        strcpy(npyToDramInstr.src_fname, sbAtomLoadWaveop->gRefFileName().c_str());
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        npyToDramInstr.dst_addr     = npyFileDramOffset;
        m_WaveCode.writeInstruction(npyToDramInstr);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomLoadWaveop->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomLoadWaveop->gRefFileShape();
        m_WaveCode.recordDramForNpyFile(sbAtomLoadWaveop->gRefFileName(), npyFileInfo);
    }


    compisa::SimDmaCopyInstr simDmaCopyInstr;
    simDmaCopyInstr.inst_events.wait_event_idx  = 0;
    simDmaCopyInstr.inst_events.wait_event_mode = eventWaitMode2Isa(events::EventWaitMode::DontWait);
    simDmaCopyInstr.inst_events.set_event_idx   = 0;
    simDmaCopyInstr.inst_events.set_event_mode  = eventSetMode2Isa(events::EventSetMode::DontSet);

    events::EventId setEventId = 0; // events::EventId_Invalid();
    events::EventSetMode setEventMode = events::EventSetMode::DontSet;
    events::EventId waitEventId = 0; // events::EventId_Invalid();
    events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(sbAtomLoadWaveop, waitEventId, waitEventMode);
    } // end incoming events

    simDmaCopyInstr.inst_events.wait_event_idx  = waitEventId;
    simDmaCopyInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(waitEventMode);

    if (qParallelStreams()) { // Find first successor for embedded
        findFirstSetEventIdMode(sbAtomLoadWaveop, setEventId,  setEventMode);
    }

    //************************************************************************
    const kcc_int64 fileAddress = npyFileDramOffset + sbAtomLoadWaveop->gOffsetInFile() + (startPart * stepSize);
    const TongaAddress sbStartTongaAddress = stateBuf.gEntryTongaAddress(startPart, addressInPart);

    // DRAM
    simDmaCopyInstr.src_start_addr   = fileAddress;
    simDmaCopyInstr.src_num_elem[0]  = numPartitions * numBytesPerPart;
    simDmaCopyInstr.src_step_elem[0] = 1;
    simDmaCopyInstr.src_num_elem[1]  = 1;
    simDmaCopyInstr.src_step_elem[1] = 0;

    // SB
    simDmaCopyInstr.dst_start_addr   = sbStartTongaAddress;
    simDmaCopyInstr.dst_num_elem[0]  = numBytesPerPart;
    simDmaCopyInstr.dst_step_elem[0] = 1;
    simDmaCopyInstr.dst_num_elem[1]  = numPartitions;
    simDmaCopyInstr.dst_step_elem[1] = stateBuf.gEntryTongaAddress(1, addressInPart) - stateBuf.gEntryTongaAddress(0, addressInPart);

    // Should we assert that size <= 1?
    if (succEventIds.size() > 0) {
        simDmaCopyInstr.queue_idx        = succEventIds[0];
    } else {
        simDmaCopyInstr.queue_idx    = 0;
    }

    m_WaveCode.writeInstruction(simDmaCopyInstr, chosenEngId);
}

//======================================================================
kcc_int32
WaveCodeSbAtomLoad::findSuccEventsAndChosenEngine(wave::SbAtomWaveOp* sbAtomWaveop,
                        EngineId& chosenEngId,
                        std::vector<events::EventId>& succEventIds)
{
    kcc_int32 numSyncs = 0;
    wave::WaveEdge* chosenPrevEdge = nullptr;

    for (auto prevWaveEdge : sbAtomWaveop->gPrevWaveEdges()) {
        if (prevWaveEdge->qChosenForSuccSbAtom()) {
            chosenPrevEdge = prevWaveEdge;
            break;
        }
    }
    if (chosenPrevEdge) {
        chosenEngId = chosenPrevEdge->gFromOp()->gEngineId();
    } else {
        chosenEngId = EngineId::Pooling;
    }

    // First wait on all other engines
    for (auto prevWaveEdge : sbAtomWaveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementSync()) {
            continue;
        }
        if (prevWaveEdge == chosenPrevEdge) {
            continue;
        }
        ++numSyncs;
        writeWaitOrWaitClearInstr(prevWaveEdge, chosenEngId);
    }

    for (auto succWaveEdge : sbAtomWaveop->gSuccWaveEdges()) {
        if (succWaveEdge->qNeedToImplementSync()) {
            succEventIds.push_back(succWaveEdge->gEventId());
            ++numSyncs;
        }
    }
    return numSyncs;
}


}}

