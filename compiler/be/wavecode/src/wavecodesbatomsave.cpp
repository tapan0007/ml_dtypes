#include "utils/inc/asserter.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"


#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisadmatrigger.hpp"
#include "compisa/inc/compisasimdmacopy.hpp"

#include "events/inc/events.hpp"

#include "layers/inc/layer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"

namespace kcc {
namespace wavecode {



//************************************************************************
WaveCodeSbAtomSave::WaveCodeSbAtomSave(WaveCodeRef waveCode)
    : WaveCodeSbAtom(waveCode)
{}

//************************************************************************
void
WaveCodeSbAtomSave::generate(wave::WaveOp* waveop)
{
    auto sbAtomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveop);
    Assert(sbAtomSaveWaveop, "Expecting Save waveop");
    calcOutputSize(sbAtomSaveWaveop);
    if (qGenerateKelf()) {
        generateForKelf(sbAtomSaveWaveop);
    } else {
        generateForSim(sbAtomSaveWaveop);
    }
}


//************************************************************************
void
WaveCodeSbAtomSave::generateForSim(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop)
{
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomSaveWaveop->gEngineId();
    Assert(EngineId::None != engineId, "Engine id for SbAtomSave waveop should not be None");

    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(sbAtomSaveWaveop->gRefFileName());

    if (npyFileDramOffset < 0) {
        const kcc_int64 numPySize = sbAtomSaveWaveop->gSaveDataSizeInBytes();
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomSaveWaveop->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomSaveWaveop->gRefFileShape();
        m_WaveCode.recordDramForNpyFile(sbAtomSaveWaveop->gRefFileName(), npyFileInfo);
    }

    compisa::SimMemCpyInstr statebufToDramInstr;

    statebufToDramInstr.inst_events.set_event_idx      = 0;
    statebufToDramInstr.inst_events.set_event_mode     = eventSetMode2Isa(events::EventSetMode::DontSet);
    statebufToDramInstr.inst_events.wait_event_idx     = 0;
    statebufToDramInstr.inst_events.wait_event_mode    = eventWaitMode2Isa(events::EventWaitMode::DontWait);

    events::EventId setEventId          = 0; // events::EventId_Invalid();
    events::EventSetMode setEventMode   = events::EventSetMode::DontSet;
    events::EventId waitEventId         = 0; // events::EventId_Invalid();
    events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

    //************************************************************************
    if (qParallelStreams()) { // Incoming edges/events: Wait for events from predecessors
        processIncomingEdges(sbAtomSaveWaveop, waitEventId, waitEventMode);
    }


    if (qParallelStreams()) { // Find first successor for embedded
        findFirstSetEventIdMode(sbAtomSaveWaveop, setEventId,  setEventMode);
    }

    //************************************************************************
    // Instruction
    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomSaveWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomSaveWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomSaveWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomSaveWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomSaveWaveop->gStartAtMidPart() ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    statebufToDramInstr.nbytes      = numBytesPerPart;

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        // TODO: add synchronization during DMA through extra DMA descriptor
        if (qParallelStreams()) {
            statebufToDramInstr.inst_events.wait_event_idx     = 0;
            statebufToDramInstr.inst_events.wait_event_mode    = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
            statebufToDramInstr.inst_events.set_event_idx      = 0;
            statebufToDramInstr.inst_events.set_event_mode     = events::eventSetMode2Isa(events::EventSetMode::DontSet);

            if (0 == partIdx) { // only the first reading waits for predecessors
                statebufToDramInstr.inst_events.wait_event_idx     = waitEventId;
                statebufToDramInstr.inst_events.wait_event_mode    = events::eventWaitMode2Isa(waitEventMode);
            }

            if (numPartitions-1 == partIdx) { // only the last reading informs successors
                statebufToDramInstr.inst_events.set_event_idx      = setEventId;
                statebufToDramInstr.inst_events.set_event_mode     = events::eventSetMode2Isa(setEventMode);

            }
        }

        statebufToDramInstr.src_addr = stateBuf.gEntryTongaAddress(partIdx, addressInPart);
        statebufToDramInstr.dst_addr = npyFileDramOffset + sbAtomSaveWaveop->gOffsetInFile() + (partIdx * stepSize);

        {
            std::ostringstream oss;
            oss << sbAtomSaveWaveop->gOrder() << "-" << sbAtomSaveWaveop->gName() << "-" <<partIdx;
            m_WaveCode.SaveName(statebufToDramInstr, oss.str().c_str());
        }
        m_WaveCode.writeInstruction(statebufToDramInstr);

        m_WaveCode.markDramDirty(sbAtomSaveWaveop->gRefFileName());
    }

    //************************************************************************
    if (qParallelStreams()) { // Write remaining SETs
        processOutgoingEdgesAlreadyEmb(sbAtomSaveWaveop, setEventId);
    }
}


//************************************************************************
void
WaveCodeSbAtomSave::generateForKelf(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop)
{
    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;
    /*const kcc_int32 numSyncs =*/ findSuccEventsAndChosenEngine(sbAtomSaveWaveop,
                                        chosenEngId, succEventIds);

    if (m_WaveCode.qBinFileSimKelf()) {
        generateDmaCopySimKelf(sbAtomSaveWaveop, chosenEngId, succEventIds);
    } else{
        // TODO: handle debug and off-load SbAtomSaves
        generateDmaTriggerRuntimeKelf(sbAtomSaveWaveop, chosenEngId, succEventIds);
    }
}



//************************************************************************
void
WaveCodeSbAtomSave::generateDmaTriggerRuntimeKelf(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());


    const kcc_int64 numPartitions   = sbAtomSaveWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomSaveWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomSaveWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomSaveWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomSaveWaveop->gStartAtMidPart() ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;

    //Assert(succEventIds.size() == 1, "AtomSave: only one succ event id");
    //************************************************************************
    //const bool qOut = sbAtomSaveWaveop->gSuccWaveEdges().size() == 0;
    const bool qOut = sbAtomSaveWaveop-> qFinalLayerOfmap();
    std::ostringstream oss;
    oss << sbAtomSaveWaveop->gOrder() << "-" << sbAtomSaveWaveop->gName();
    kelf::DmaDescription::DmaBlockFromTpb& dmaBlock(kelfDma.startNewDmaBlockFromTpb(
                                        chosenEngId, qOut, oss.str().c_str()));
    const std::string refFileName(sbAtomSaveWaveop->gRefFileName());

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        const TongaAddress  fileAddress = sbAtomSaveWaveop->gOffsetInFile() + (partIdx * stepSize);
        const TpbAddress    sbAddress = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        dmaBlock.addDmaDesc(sbAddress, fileAddress, refFileName, numBytesPerPart);
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

        processIncomingEdgesForceWait(sbAtomSaveWaveop, chosenEngId, waitEventId, waitEventMode);
    } // end incoming events

    //************************************************************************
    addDmaBarrier(chosenEngId);
    //************************************************************************
    compisa::DmaTriggerInstr dmaTriggerInstr;
    strncpy(dmaTriggerInstr.dma_queue_name, dmaBlock.gQueueName().c_str(),
            sizeof(dmaTriggerInstr.dma_queue_name)/sizeof(dmaTriggerInstr.dma_queue_name[0]) - 1);

    dmaTriggerInstr.use_raw_count = 0; // get from JSON
    dmaTriggerInstr.block_id = dmaBlock.gBlockId();

    dmaTriggerInstr.inst_events.wait_event_idx  = 0;
    dmaTriggerInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    dmaTriggerInstr.inst_events.set_event_idx   = 0; // succ event is in desc
    dmaTriggerInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    {
        std::ostringstream oss;
        if (succEventIds.size() > 0)
            oss << sbAtomSaveWaveop->gOrder() << ":" << succEventIds[0] << "-" << sbAtomSaveWaveop->gName();
        else
            oss << sbAtomSaveWaveop->gOrder() << ":-1" << "-" << sbAtomSaveWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
    addSecondDmaTrigger(dmaTriggerInstr, chosenEngId);
}






//************************************************************************
void
WaveCodeSbAtomSave::generateDmaCopySimKelf(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(m_WaveCode.qBinFileSimKelf(), "Must be binary for SIM Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    const kcc_int64 numPartitions   = sbAtomSaveWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomSaveWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomSaveWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomSaveWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomSaveWaveop->gStartAtMidPart() ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;

    compisa::SimDmaCopyInstr simDmaCopyInstr;

    const TongaAddress sbStartTongaAddress = stateBuf.gEntryTongaAddress(startPart, addressInPart);
    const kcc_int64 fileAddress = sbAtomSaveWaveop->gOffsetInFile() + (startPart * stepSize);

    // SB
    simDmaCopyInstr.src_start_addr   = sbStartTongaAddress;
    simDmaCopyInstr.src_num_elem[0]  = numBytesPerPart;
    simDmaCopyInstr.src_step_elem[0] = 1;
    simDmaCopyInstr.src_num_elem[1]  = numPartitions;
    simDmaCopyInstr.src_step_elem[1] = stepSize;

    // DRAM
    simDmaCopyInstr.dst_start_addr   = fileAddress;
    simDmaCopyInstr.dst_num_elem[0]  = numPartitions * numBytesPerPart;
    simDmaCopyInstr.dst_step_elem[0] = 1;
    simDmaCopyInstr.dst_num_elem[1]  = 1;
    simDmaCopyInstr.dst_step_elem[1] = 0;

    // Should we assert that size <= 1?
    if (succEventIds.size() > 0) {
        simDmaCopyInstr.queue_idx    = succEventIds[0];
    } else {
        simDmaCopyInstr.queue_idx    = 0;
    }

    m_WaveCode.writeInstruction(simDmaCopyInstr, chosenEngId);
}

//======================================================================
kcc_int32
WaveCodeSbAtomSave::findSuccEventsAndChosenEngine(wave::SbAtomSaveWaveOp* sbAtomWaveop,
                        EngineId& chosenEngId,
                        std::vector<events::EventId>& succEventIds)
{
    chosenEngId = sbAtomWaveop->gEngineId();
    Assert(chosenEngId != EngineId::None, "None engine in waveop ", sbAtomWaveop->gName());
    kcc_int32 numSyncs = 0;
    wave::WaveEdge* chosenPrevEdge = nullptr;

    for (auto prevWaveEdge : sbAtomWaveop->gPrevWaveEdges()) {
        if (prevWaveEdge->qChosenForSuccSbAtom()) {
            chosenPrevEdge = prevWaveEdge;
            break;
        }
    }
    Assert(chosenPrevEdge, "Save: must have prev chosen edge");
    Assert(chosenEngId == chosenPrevEdge->gFromOp()->gEngineId(),
        "Engine on chosen edge from ", chosenPrevEdge->gFromOp()->gName(), " to ", sbAtomWaveop->gName(),
        " different than engine id ", utils::engineId2Str(chosenEngId));

    // First wait on all other engines
    for (auto prevWaveEdge : sbAtomWaveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementSync()) {
            continue;
        }
        if (prevWaveEdge == chosenPrevEdge) {
            continue;
        }
        ++numSyncs;
        m_WaveCode.writeWaitOrWaitClearInstr(prevWaveEdge, chosenEngId);
    }
    for (auto succWaveEdge : sbAtomWaveop->gSuccWaveEdges()) {
        if (succWaveEdge->qNeedToImplementSync()) {
            succEventIds.push_back(succWaveEdge->gEventId());
            ++numSyncs;
        }
    }
    return numSyncs;
}

void
WaveCodeSbAtomSave::calcOutputSize(const wave::SbAtomSaveWaveOp* sbAtomSaveWaveop)
{
    const bool qOut = sbAtomSaveWaveop-> qFinalLayerOfmap();
    if (!qOut) {
        return;
    }
    const utils::DataType&    dtype(sbAtomSaveWaveop->gDataType());
    const std::array<kcc_int32,4>& shape(sbAtomSaveWaveop->gRefFileShape ());
    kcc_int64 sz = dtype.gSizeInBytes();
    for (auto n : shape) {
        sz *= n;
    }
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    if (kelfDma.gOutputSizeBytes() < 0) {
        kelfDma.rOutputSizeBytes(sz);
    } else {
        if (m_WaveCode.qBinFileRuntimeKelf()) {
            Assert(kelfDma.gOutputSizeBytes() == sz,
                "Previously calculated output size ", kelfDma.gOutputSizeBytes(),
                " != current size ", sz, " from AtomSave ",
                sbAtomSaveWaveop->gName());
        }
    }
}

}}


