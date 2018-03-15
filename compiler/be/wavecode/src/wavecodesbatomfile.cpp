#include "shared/inc/tpb_isa_ldweights.hpp"
#include "shared/inc/tpb_isa_write.hpp"


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
    Assert(EngineId::DmaEng == engineId, "Engine id for SbAtomFile waveop should be DmaEng, but is ",
           static_cast<long>(engineId));

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

    //************************************************************************
    { // Outgoing events: inform successors
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
            Assert(false, "sbAtomFile waveop ", sbAtomFileWaveOp->gName(), ": successor waveop ", succWaveop->gName(),
                   " has wrong type ", succWaveop->gTypeStr());
        }

        //************************************************************************
        // Find one embedded set-event edge, if any
        //************************************************************************
        const wave::WaveEdge* succWaveEdgeEmb  = nullptr;

        kcc_uint32 matmulStart = 0;
        if (!succWaveEdgeEmb && succMatmulEdges.size() > 0) {
            succWaveEdgeEmb = succMatmulEdges[matmulStart++];
        }
        kcc_uint32 activationStart = 0;
        if (!succWaveEdgeEmb && succActivationEdges.size() > 0) {
            succWaveEdgeEmb = succActivationEdges[activationStart++];
        }
        kcc_uint32 poolStart = 0;
        if (!succWaveEdgeEmb && succPoolEdges.size() > 0) {
            succWaveEdgeEmb = succPoolEdges[poolStart++];
        }
        Assert(matmulStart + activationStart + poolStart == 1, "SbAtomFile ", sbAtomFileWaveOp->gName(),
               " must have exactly one successor. Number successors: MatMul=", succMatmulEdges.size(),
               ", Activation=", succActivationEdges.size(), ", Pool=", succPoolEdges.size());
        Assert(succWaveEdgeEmb, "SbAtomFile must have at least one successor");

        const EventId setEventId = succWaveEdgeEmb->gEventId();
        const events::EventSetMode eventSetMode = succWaveEdgeEmb->gSetEventMode();


        //************************************************************************
        // Instruction(s)
        //************************************************************************
        const kcc_int64 numPartitions   = sbAtomFileWaveOp->gIfmapCount();
        const kcc_int64 numBytesPerPart = sbAtomFileWaveOp->gLength();
        const kcc_int64 addressInPart   = sbAtomFileWaveOp->gAddressInPartition(0 /*offset in atom*/);
        const kcc_int64 stepSize = sbAtomFileWaveOp->gPartitionStepBytes();

        SIM_MEMCPY dramToStateBufInstr;
        dramToStateBufInstr.nbytes                  = numBytesPerPart;
        dramToStateBufInstr.sync.wait_event_id      = -1;
        dramToStateBufInstr.sync.wait_event_mode    = eventWaitMode2Int(events::EventWaitMode::NoEvent);

        for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
            if (numPartitions-1 == partIdx) {
                // only the last reading sets event to enable subsequent instr
                dramToStateBufInstr.sync.set_event_id       = setEventId;
                dramToStateBufInstr.sync.set_event_mode     = events::eventSetMode2Int(eventSetMode);
            } else {
                dramToStateBufInstr.sync.set_event_id       = -1;
                dramToStateBufInstr.sync.set_event_mode     = eventSetMode2Int(events::EventSetMode::NoEvent);
            }

            dramToStateBufInstr.src_address = npyFileDramOffset + sbAtomFileWaveOp->gOffsetInFile() + (partIdx * stepSize);
            dramToStateBufInstr.dst_address = stateBuf.gEntrySysAddress(partIdx, addressInPart);

            m_WaveCode->writeInstruction(dramToStateBufInstr);
        }
        //************************************************************************
    } 
}

}}

