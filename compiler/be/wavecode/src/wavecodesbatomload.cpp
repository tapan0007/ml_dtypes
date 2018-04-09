#include "compisa/inc/compisaldweights.hpp"
#include "compisa/inc/compisawrite.hpp"
#include "compisa/inc/compisasimwrnpy.hpp"
#include "compisa/inc/compisasimmemcpy.hpp"


#include "utils/inc/debug.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/convlayer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomload.hpp"

namespace kcc {
namespace wavecode {




WaveCodeSbAtomLoad::WaveCodeSbAtomLoad(WaveCodeRef waveCode)
    : WaveCodeSbAtom(waveCode)
{}


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
WaveCodeSbAtomLoad::generate(wave::WaveOp* waveOp)
{
    const auto sbAtomLoadWaveOp = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveOp);
    assert(sbAtomLoadWaveOp);
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomLoadWaveOp->gEngineId();
    Assert(EngineId::DmaEng == engineId, "Engine id for SbAtomLoad waveop should be DmaEng, but is ",
           static_cast<long>(engineId));

    //************************************************************************
    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(sbAtomLoadWaveOp->gRefFileName());
    if (npyFileDramOffset < 0) { // Load whole numpy file to DRAM
        compisa::SimWrNpyInstr npyToDramInstr;
        npyToDramInstr.sync.wait_event_id      = 0;
        npyToDramInstr.sync.wait_event_mode    = eventWaitMode2Int(events::EventWaitMode::DontWait);
        npyToDramInstr.sync.set_event_id      = 0;
        npyToDramInstr.sync.set_event_mode    = eventSetMode2Int(events::EventSetMode::DontSet);

        const kcc_int64 numPySize = sbAtomLoadWaveOp->gLoadDataSizeInBytes();
        strcpy(npyToDramInstr.src_fname, sbAtomLoadWaveOp->gRefFileName().c_str());
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        npyToDramInstr.dst_address  = npyFileDramOffset;
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
    dramToStateBufInstr.sync.wait_event_id      = 0;
    dramToStateBufInstr.sync.wait_event_mode    = eventWaitMode2Int(events::EventWaitMode::DontWait);
    dramToStateBufInstr.sync.set_event_id       = 0;
    dramToStateBufInstr.sync.set_event_mode     = eventSetMode2Int(events::EventSetMode::DontSet);

    events::EventId setEventId = 0; // events::EventId_Invalid();
    events::EventSetMode setEventMode = events::EventSetMode::DontSet;
    events::EventId waitEventId = 0; // events::EventId_Invalid();
    events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(sbAtomLoadWaveOp, waitEventId, waitEventMode);
    } // end incoming events


    if (qParallelStreams()) { // Find first successor for embedded
        findSetEventIdMode(sbAtomLoadWaveOp, setEventId,  setEventMode);
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
            dramToStateBufInstr.sync.wait_event_id      = 0;
            dramToStateBufInstr.sync.wait_event_mode    = events::eventWaitMode2Int(events::EventWaitMode::DontWait);
            dramToStateBufInstr.sync.set_event_id       = 0;
            dramToStateBufInstr.sync.set_event_mode     = events::eventSetMode2Int(events::EventSetMode::DontSet);

            if (0 == partIdx) { // only the first reading waits for predecessors
                dramToStateBufInstr.sync.wait_event_id      = waitEventId;
                dramToStateBufInstr.sync.wait_event_mode    = events::eventWaitMode2Int(waitEventMode);
            }

            if (numPartitions-1 == partIdx) { // only the last reading informs successors
                dramToStateBufInstr.sync.set_event_id       = setEventId;
                dramToStateBufInstr.sync.set_event_mode     = events::eventSetMode2Int(setEventMode);
            }
        }

        dramToStateBufInstr.src_address = npyFileDramOffset + sbAtomLoadWaveOp->gOffsetInFile() + (partIdx * stepSize);
        dramToStateBufInstr.dst_address = stateBuf.gEntrySysAddress(partIdx, addressInPart);

        m_WaveCode.writeInstruction(dramToStateBufInstr);
    }

    //************************************************************************
    if (qParallelStreams()) { // Write remaining SETs
        processOutgoingEdgesAlreadyEmb(sbAtomLoadWaveOp);
    }
}

}}

