#include "shared/inc/tpb_isa_ldweights.hpp"


#include "events/inc/events.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "layers/inc/layer.hpp"

#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtomSave::WaveCodeSbAtomSave(WaveCode* waveCode)
    : WaveCodeSbAtom(waveCode)
{}

void
WaveCodeSbAtomSave::generate(wave::WaveOp* waveOp)
{
    auto sbAtomSaveWaveOp = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveOp);
    assert(sbAtomSaveWaveOp);
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    kcc_int64 npyFileDramOffset = m_WaveCode->getDramForNpyFile(sbAtomSaveWaveOp->gRefFileName());

    if (npyFileDramOffset < 0) {
        const kcc_int64 numPySize = sbAtomSaveWaveOp->gSaveDataSizeInBytes();
        npyFileDramOffset           = m_WaveCode->gCurrentDramAddress(numPySize);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomSaveWaveOp->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomSaveWaveOp->gRefFileShape();
        m_WaveCode->recordDramForNpyFile(sbAtomSaveWaveOp->gRefFileName(), npyFileInfo);
    }

    const kcc_int64 numPartitions   = sbAtomSaveWaveOp->gOfmapCount();
    const kcc_int64 numBytesPerPart = sbAtomSaveWaveOp->gLength();
    const kcc_int64 addressInPart   = sbAtomSaveWaveOp->gAddressInPartition(0 /*offset in atom*/);
    const kcc_int64 stepSize = sbAtomSaveWaveOp->gPartitionStepBytes();

    SIM_MEMCPY statebufToDramInstr;
    statebufToDramInstr.nbytes       = numBytesPerPart;
    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {

        statebufToDramInstr.sync.set_event_id       = -1;
        statebufToDramInstr.sync.set_event_mode     = eventSetMode2Int(events::EventSetMode::NoEvent);
        statebufToDramInstr.sync.wait_event_id      = -1;
        statebufToDramInstr.sync.wait_event_mode    = eventWaitMode2Int(events::EventWaitMode::NoEvent);
        if (0 == partIdx) {
            // only the first reading waits for events from previous instr
            statebufToDramInstr.sync.wait_event_id      = sbAtomSaveWaveOp->gWaitEventId();
            statebufToDramInstr.sync.wait_event_mode    = events::WaitEvent::eventWaitMode2Int(sbAtomSaveWaveOp->gWaitEventMode());
        }
        if (numPartitions-1 == partIdx) {
            // only the last reading sets event to enable subsequent instr
            statebufToDramInstr.sync.set_event_id       = sbAtomSaveWaveOp->gSetEventId();
            statebufToDramInstr.sync.set_event_mode     = events::SetEvent::eventSetMode2Int(sbAtomSaveWaveOp->gSetEventMode());
        }

        statebufToDramInstr.src_address = stateBuf.gEntrySysAddress(partIdx, addressInPart);
        statebufToDramInstr.dst_address = npyFileDramOffset + sbAtomSaveWaveOp->gOffsetInFile() + (partIdx * stepSize);
        m_WaveCode->writeInstruction(statebufToDramInstr);
        m_WaveCode->markDramDirty(sbAtomSaveWaveOp->gRefFileName());
    }
}


}}


