#include "shared/inc/tpb_isa_ldweights.hpp"


#include "events/inc/events.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/convlayer.hpp"

#include "wave/inc/sbatomfilewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomfile.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtomFile::WaveCodeSbAtomFile(WaveCode* waveCode)
    : WaveCodeSbAtom(waveCode)
{}

void
WaveCodeSbAtomFile::generate(wave::WaveOp* waveOp)
{
    const auto sbAtomFileWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(waveOp);
    assert(sbAtomFileWaveOp);
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    kcc_int64 npyFileDramOffset = m_WaveCode->getDramForNpyFile(sbAtomFileWaveOp->gRefFileName());
    if (npyFileDramOffset < 0) {
        SIM_WRNPY npyToDramInstr;
        // Load whole numpy file
        const kcc_int64 numPySize = sbAtomFileWaveOp->gLoadDataSizeInBytes();
        strcpy(npyToDramInstr.src_fname, sbAtomFileWaveOp->gRefFileName().c_str());
        npyFileDramOffset           = m_WaveCode->gCurrentDramAddress(numPySize);

        npyToDramInstr.dst_address  = npyFileDramOffset;
        m_WaveCode->writeInstruction(npyToDramInstr);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomFileWaveOp->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomFileWaveOp->gRefFileShape();
        m_WaveCode->recordDramForNpyFile(sbAtomFileWaveOp->gRefFileName(), npyFileInfo);
    }

    const kcc_int64 numPartitions   = sbAtomFileWaveOp->gIfmapCount();
    const kcc_int64 numBytesPerPart = sbAtomFileWaveOp->gLength();
    const kcc_int64 addressInPart   = sbAtomFileWaveOp->gAddressInPartition(0 /*offset in atom*/);
    const kcc_int64 stepSize = sbAtomFileWaveOp->gPartitionStepBytes();

    SIM_MEMCPY dramToStateBufInstr;
    dramToStateBufInstr.nbytes = numBytesPerPart;
    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {

        dramToStateBufInstr.sync.set_event_id       = -1;
        dramToStateBufInstr.sync.set_event_mode     = eventSetMode2Int(events::EventSetMode::NoEvent);
        dramToStateBufInstr.sync.wait_event_id      = -1;
        dramToStateBufInstr.sync.wait_event_mode    = eventWaitMode2Int(events::EventWaitMode::NoEvent);

        if (0 == partIdx) {
            // only the first reading waits for events from previous instr
            dramToStateBufInstr.sync.wait_event_id      = sbAtomFileWaveOp->gWaitEventId();
            dramToStateBufInstr.sync.wait_event_mode    = events::WaitEvent::eventWaitMode2Int(sbAtomFileWaveOp->gWaitEventMode());
        }
        if (numPartitions-1 == partIdx) {
            // only the last reading sets event to enable subsequent instr
            dramToStateBufInstr.sync.set_event_id       = sbAtomFileWaveOp->gSetEventId();
            dramToStateBufInstr.sync.set_event_mode     = events::SetEvent::eventSetMode2Int(sbAtomFileWaveOp->gSetEventMode());
        }

        dramToStateBufInstr.src_address = npyFileDramOffset + sbAtomFileWaveOp->gOffsetInFile() + (partIdx * stepSize);
        dramToStateBufInstr.dst_address = stateBuf.gEntrySysAddress(partIdx, addressInPart);
        m_WaveCode->writeInstruction(dramToStateBufInstr);
    }
}

}}

