#include "utils/inc/asserter.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"


#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisasimdmacopy.hpp"

#include "events/inc/events.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomsave_sim.hpp"

namespace kcc {
namespace wavecode {


//************************************************************************
WaveCodeSbAtomSaveSim::WaveCodeSbAtomSaveSim(WaveCodeRef waveCode)
    : WaveCodeSbAtomSave(waveCode)
{}

//************************************************************************
void
WaveCodeSbAtomSaveSim::generate(wave::WaveOp* waveop)
{
    Assert(!qGenerateKelf(), "Must be in Sim mode to save for Sim");
    auto sbAtomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveop);
    Assert(sbAtomSaveWaveop, "Expecting Save waveop");
    calcOutputSize(sbAtomSaveWaveop);
    generateForSim(sbAtomSaveWaveop);
}




//************************************************************************
void
WaveCodeSbAtomSaveSim::generateForSim(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop)
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

}}


