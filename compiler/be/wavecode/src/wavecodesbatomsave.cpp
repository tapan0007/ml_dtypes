
#include "compisa/inc/compisasimmemcpy.hpp"

#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "layers/inc/layer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"

namespace kcc {
namespace wavecode {



WaveCodeSbAtomSave::WaveCodeSbAtomSave(WaveCodeRef waveCode)
    : WaveCodeSbAtom(waveCode)
{}



void
WaveCodeSbAtomSave::generate(wave::WaveOp* waveop)
{
    auto sbAtomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveop);
    assert(sbAtomSaveWaveop);
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomSaveWaveop->gEngineId();
    Assert(EngineId::DmaEng == engineId, "Engine id for SbAtomSave waveop should be DmaEng");

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
        findSetEventIdMode(sbAtomSaveWaveop, setEventId,  setEventMode);
    }

    //************************************************************************
    // Instruction
    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomSaveWaveop->gOfmapCount();
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

        statebufToDramInstr.src_addr = stateBuf.gEntrySysAddress(partIdx, addressInPart);
        statebufToDramInstr.dst_addr = npyFileDramOffset + sbAtomSaveWaveop->gOffsetInFile() + (partIdx * stepSize);
        m_WaveCode.writeInstruction(statebufToDramInstr);
        m_WaveCode.markDramDirty(sbAtomSaveWaveop->gRefFileName());
    }

    //************************************************************************
    if (qParallelStreams()) { // Write remaining SETs
        processOutgoingEdgesAlreadyEmb(sbAtomSaveWaveop, setEventId);
    }
}


}}


