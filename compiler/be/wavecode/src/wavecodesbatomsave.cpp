#include "utils/inc/asserter.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"


#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisadmatrigger.hpp"
#include "compisa/inc/compisasimdmacopy.hpp"

#include "events/inc/events.hpp"

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

    compisa::SimMemCpyInstr simStatebufToDramInstr;

    AssignWithSizeCheck(simStatebufToDramInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(simStatebufToDramInstr.inst_events.set_event_mode, eventSetMode2Isa(events::EventSetMode::DontSet));
    AssignWithSizeCheck(simStatebufToDramInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(simStatebufToDramInstr.inst_events.wait_event_mode, eventWaitMode2Isa(events::EventWaitMode::DontWait));

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
    AssignWithSizeCheck(simStatebufToDramInstr.nbytes, numBytesPerPart);

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        // TODO: add synchronization during DMA through extra DMA descriptor
        if (qParallelStreams()) {
            AssignWithSizeCheck(simStatebufToDramInstr.inst_events.wait_event_idx, 0);
            AssignWithSizeCheck(simStatebufToDramInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
            AssignWithSizeCheck(simStatebufToDramInstr.inst_events.set_event_idx, 0);
            AssignWithSizeCheck(simStatebufToDramInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

            if (0 == partIdx) { // only the first reading waits for predecessors
                AssignWithSizeCheck(simStatebufToDramInstr.inst_events.wait_event_idx, waitEventId);
                AssignWithSizeCheck(simStatebufToDramInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(waitEventMode));
            }

            if (numPartitions-1 == partIdx) { // only the last reading informs successors
                AssignWithSizeCheck(simStatebufToDramInstr.inst_events.set_event_idx, setEventId);
                AssignWithSizeCheck(simStatebufToDramInstr.inst_events.set_event_mode, events::eventSetMode2Isa(setEventMode));

            }
        }

        AssignWithSizeCheck(simStatebufToDramInstr.src_addr, stateBuf.gEntryTongaAddress(partIdx, addressInPart));
        AssignWithSizeCheck(simStatebufToDramInstr.dst_addr, npyFileDramOffset + sbAtomSaveWaveop->gOffsetInFile() + (partIdx * stepSize));

        {
            std::ostringstream oss;
            oss << sbAtomSaveWaveop->gOrder() << "-" << sbAtomSaveWaveop->gName() << "-" <<partIdx;
            m_WaveCode.SaveName(simStatebufToDramInstr, oss.str().c_str());
        }
        m_WaveCode.writeInstruction(simStatebufToDramInstr);

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

    // TODO: handle debug and off-load SbAtomSaves
    generateDmaTriggerRuntimeKelf(sbAtomSaveWaveop, chosenEngId, succEventIds);
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

    //************************************************************************
    const bool qOut = sbAtomSaveWaveop-> qFinalLayerOfmap();
    std::ostringstream oss;
    oss << sbAtomSaveWaveop->gOrder() << "-" << sbAtomSaveWaveop->gName();

    const kcc_int32 blockIdx = kelfDma.startNewDmaBlockFromTpb(sbAtomSaveWaveop->gDmaQueue(), chosenEngId, qOut, oss.str().c_str());
    kelf::DmaDescription::DmaBlockFromTpb* dmaBlock = kelfDma.gDmaBlockFromTpb(blockIdx);

    const std::string refFileName(sbAtomSaveWaveop->gRefFileName());

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        const TongaAddress  fileAddress = sbAtomSaveWaveop->gOffsetInFile() + (partIdx * stepSize);
        const TpbAddress    sbAddress = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        dmaBlock->addDmaDesc(sbAddress, fileAddress, refFileName, numBytesPerPart);
    }
    for (auto eventId : succEventIds) {
        dmaBlock->addTailEventId(eventId);
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
    addDmaBarrier(sbAtomSaveWaveop, chosenEngId);
    //************************************************************************
    compisa::DmaTriggerInstr dmaTriggerInstr;
    dmaTriggerInstr.SetDmaQueueName(dmaBlock->gDmaQueue()->gName().c_str());

    AssignWithSizeCheck(dmaTriggerInstr.use_raw_count, 0); // get from JSON
    AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaBlock->gBlockId());

    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_idx, 0); // succ event is in desc
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));
    {
        const auto eventId = (succEventIds.size() > 0 ?  succEventIds[0] : -1);
        std::ostringstream oss;
        oss << sbAtomSaveWaveop->gOrder() << ":" << eventId << "-" << sbAtomSaveWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
}






void
WaveCodeSbAtomSave::calcOutputSize(const wave::SbAtomSaveWaveOp* sbAtomSaveWaveop)
{
    const utils::DataType&    dtype(sbAtomSaveWaveop->gDataType());
    const utils::TensorParams::ShapeType& shape(sbAtomSaveWaveop->gRefFileShape ());
    kcc_int64 sz = dtype.gSizeInBytes();
    for (auto n : shape) {
        sz *= n;
    }
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    const kcc_int64 existingSz = kelfDma.gOutputSizeBytes(sbAtomSaveWaveop->gRefFileName());
    if (existingSz < 0) {
        kelfDma.rOutputSizeBytes(sz, sbAtomSaveWaveop->gRefFileName());
    } else {
        if (m_WaveCode.qBinFileRuntimeKelf()) {
            Assert(existingSz == sz,
                "Previously calculated output size ", existingSz,
                " != current size ", sz, " from AtomSave ",
                sbAtomSaveWaveop->gName());
        }
    }
}


//************************************************************************

}}


