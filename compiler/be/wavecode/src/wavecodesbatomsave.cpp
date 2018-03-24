#include "shared/inc/tpb_isa_ldweights.hpp"


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
#define ASSERT_HAS_EVENT(edge, from, to) Assert((edge)->gEventId() != EventId_Invalid, "WaveEdge from waveop ", \
            (from)->gName(), " to waveop ", (to)->gName(), " has no event")



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

    SIM_MEMCPY statebufToDramInstr;

    statebufToDramInstr.sync.set_event_id       = -1;
    statebufToDramInstr.sync.set_event_mode     = eventSetMode2Int(events::EventSetMode::NoEvent);
    statebufToDramInstr.sync.wait_event_id      = -1;
    statebufToDramInstr.sync.wait_event_mode    = eventWaitMode2Int(events::EventWaitMode::NoEvent);

    EventId setEventId = -1;
    events::EventSetMode setEventMode = events::EventSetMode::NoEvent;
    EventId waitEventId = -1;
    events::EventWaitMode waitEventMode = events::EventWaitMode::NoEvent;

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
    statebufToDramInstr.nbytes      = numBytesPerPart;

    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
        // TODO: add synchronization during DMA through extra DMA descriptor
        if (qParallelStreams()) {
            statebufToDramInstr.sync.wait_event_id      = -1;
            statebufToDramInstr.sync.wait_event_mode    = events::eventWaitMode2Int(events::EventWaitMode::NoEvent);
            statebufToDramInstr.sync.set_event_id       = -1;
            statebufToDramInstr.sync.set_event_mode     = events::eventSetMode2Int(events::EventSetMode::NoEvent);

            if (0 == partIdx) { // only the first reading waits for predecessors
                statebufToDramInstr.sync.wait_event_id      = waitEventId;
                statebufToDramInstr.sync.wait_event_mode    = events::eventWaitMode2Int(waitEventMode);
            }

            if (numPartitions-1 == partIdx) { // only the last reading informs successors
                statebufToDramInstr.sync.set_event_id       = setEventId;
                statebufToDramInstr.sync.set_event_mode     = events::eventSetMode2Int(setEventMode);

            }
        }

        statebufToDramInstr.src_address = stateBuf.gEntrySysAddress(partIdx, addressInPart);
        statebufToDramInstr.dst_address = npyFileDramOffset + sbAtomSaveWaveop->gOffsetInFile() + (partIdx * stepSize);
        m_WaveCode.writeInstruction(statebufToDramInstr);
        m_WaveCode.markDramDirty(sbAtomSaveWaveop->gRefFileName());
    }

    //************************************************************************
    if (qParallelStreams()) { // Write remaining SETs
        processOutgoingEdgesAlreadyEmb(sbAtomSaveWaveop);
    }
}


}}


