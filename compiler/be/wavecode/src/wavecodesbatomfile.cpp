#include "shared/inc/tpb_isa_ldweights.hpp"


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
#include "wave/inc/sbatomfilewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomfile.hpp"

namespace kcc {
namespace wavecode {

#define ASSERT_HAS_EVENT(edge, from, to) Assert((edge)->gEventId() != EventId_Invalid, "WaveEdge from waveop ", \
            (from)->gName(), " to waveop ", (to)->gName(), " has no event")

WaveCodeSbAtomFile::WaveCodeSbAtomFile(WaveCode* waveCode)
    : WaveCodeSbAtom(waveCode)
{}

void
WaveCodeSbAtomFile::generate(wave::WaveOp* waveOp)
{
    const auto sbAtomFileWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(waveOp);
    assert(sbAtomFileWaveOp);
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomFileWaveOp->gEngineId();

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
    dramToStateBufInstr.nbytes                  = numBytesPerPart;
    dramToStateBufInstr.sync.set_event_id       = -1;
    dramToStateBufInstr.sync.set_event_mode     = eventSetMode2Int(events::EventSetMode::NoEvent);
    dramToStateBufInstr.sync.wait_event_id      = -1;
    dramToStateBufInstr.sync.wait_event_mode    = eventWaitMode2Int(events::EventWaitMode::NoEvent);
    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
#if 0
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
#endif

        dramToStateBufInstr.src_address = npyFileDramOffset + sbAtomFileWaveOp->gOffsetInFile() + (partIdx * stepSize);
        dramToStateBufInstr.dst_address = stateBuf.gEntrySysAddress(partIdx, addressInPart);
        m_WaveCode->writeInstruction(dramToStateBufInstr);
    }

    //************************************************************************

    //************************************************************************
    { // Outgoing events
        std::vector<const wave::WaveEdge*> succMatmulEdges;
        std::vector<const wave::WaveEdge*> succPoolEdges;
        std::vector<const wave::WaveEdge*> succActivationEdges;

        for (auto succWaveEdge : sbAtomFileWaveOp->gSuccWaveEdges()) {
            if (succWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto succWaveop = succWaveEdge->gToOp();
            if (succWaveop->gEngineId() == engineId) {
                continue;
            }

            if (auto succMatmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, sbAtomFileWaveOp, succMatmulWaveop);
                succMatmulEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, sbAtomFileWaveOp, succPoolWaveop);
                succPoolEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succActivationWaveop = dynamic_cast<wave::ActivationWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, sbAtomFileWaveOp, succActivationWaveop);
                succActivationEdges.push_back(succWaveEdge);
                continue;
            }
            Assert(false, "sbAtomFile waveop: successor waveop ", succWaveop->gName(), " has wrong type ", succWaveop->gTypeStr());
        }


        // Add one more MEMCPY to send events
        const utils::DataTypeUint16 dtype;
        SIM_MEMCPY writeEventInstr;
        writeEventInstr.nbytes = 1;
        writeEventInstr.sync.wait_event_id   = -1;
        writeEventInstr.sync.wait_event_mode = eventWaitMode2Int(events::EventWaitMode::NoEvent);
        writeEventInstr.src_address = stateBuf.gAllOneOffsetSysAddress(dtype);


        for (auto succWaveEdge : succPoolEdges) {
            writeEventInstr.dst_address  = m_WaveCode->calculateEventAddress(EngineId::Pooling, succWaveEdge->gEventId());
            m_WaveCode->writeInstruction(writeEventInstr, EngineId::Pooling);
        }
        for (auto succWaveEdge : succMatmulEdges) {
            writeEventInstr.dst_address  = m_WaveCode->calculateEventAddress(EngineId::PeArray, succWaveEdge->gEventId());
            m_WaveCode->writeInstruction(writeEventInstr, EngineId::PeArray);
        }
        for (auto succWaveEdge : succActivationEdges) {
            writeEventInstr.dst_address  = m_WaveCode->calculateEventAddress(EngineId::Activation, succWaveEdge->gEventId());
            m_WaveCode->writeInstruction(writeEventInstr, EngineId::PeArray);
        }
    }

    //************************************************************************

    //************************************************************************
}

}}

