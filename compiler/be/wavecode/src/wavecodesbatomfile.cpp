#include "shared/inc/tpb_isa_ldweights.hpp"
#include "shared/inc/tpb_isa_write.hpp"


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
#include "wave/inc/sbatomfilewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomfile.hpp"

namespace kcc {
namespace wavecode {

#define ASSERT_HAS_EVENT(edge, from, to) Assert((edge)->gEventId() != EventId_Invalid, "WaveEdge from waveop ", \
            (from)->gName(), " to waveop ", (to)->gName(), " has no event")



WaveCodeSbAtomFile::WaveCodeSbAtomFile(WaveCodeRef waveCode)
    : WaveCodeSbAtom(waveCode)
{}



void
WaveCodeSbAtomFile::generate(wave::WaveOp* waveOp)
{
    if (waveOp->gName() == "input/SBAtomFile_ifmaps_0_n0_m0_h5_w0_c0_r5_s0") {
        utils::breakFunc(1);
    }
    if (waveOp->gName() == "1conv/i1/MatMul_n0_m0_h0_w0_c0_r0_s0") {
        utils::breakFunc(2);
    }

    const auto sbAtomFileWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(waveOp);
    assert(sbAtomFileWaveOp);
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomFileWaveOp->gEngineId();
    Assert(EngineId::DmaEng == engineId, "Engine id for SbAtomFile waveop should be DmaEng, but is ",
           static_cast<long>(engineId));

    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(sbAtomFileWaveOp->gRefFileName());
    if (npyFileDramOffset < 0) {
        SIM_WRNPY npyToDramInstr;
        // Load whole numpy file
        const kcc_int64 numPySize = sbAtomFileWaveOp->gLoadDataSizeInBytes();
        strcpy(npyToDramInstr.src_fname, sbAtomFileWaveOp->gRefFileName().c_str());
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        npyToDramInstr.dst_address  = npyFileDramOffset;
        m_WaveCode.writeInstruction(npyToDramInstr);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomFileWaveOp->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomFileWaveOp->gRefFileShape();
        m_WaveCode.recordDramForNpyFile(sbAtomFileWaveOp->gRefFileName(), npyFileInfo);
    }

    SIM_MEMCPY dramToStateBufInstr;
    dramToStateBufInstr.sync.wait_event_id      = -1;
    dramToStateBufInstr.sync.wait_event_mode    = eventWaitMode2Int(events::EventWaitMode::NoEvent);
    dramToStateBufInstr.sync.set_event_id       = -1;
    dramToStateBufInstr.sync.set_event_mode     = eventSetMode2Int(events::EventSetMode::NoEvent);
    EventId setEventId = -1;
    events::EventSetMode eventSetMode = events::EventSetMode::NoEvent;

    //************************************************************************
    if (qParallelStreams()) { // Outgoing events: inform successors
        std::vector<const wave::WaveEdge*> succMatmulEdges;
        std::vector<const wave::WaveEdge*> succPoolEdges;
        std::vector<const wave::WaveEdge*> succActivationEdges;
        std::vector<const wave::WaveEdge*> succEdgesWithoutEvent;

        for (auto succWaveEdge : sbAtomFileWaveOp->gSuccWaveEdges()) {
            auto succWaveop = succWaveEdge->gToOp();
            if (succWaveop->gEngineId() == engineId) {
                continue;
            }
            if (succWaveEdge->gEventId() == EventId_Invalid) {
                succEdgesWithoutEvent.push_back(succWaveEdge);
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
        switch (matmulStart + activationStart + poolStart) {
        case 1: // all is good
            break;
        case 0:
            // this is one of the sb atom file, but was not chosen for the event because it is not last
            // check that there is another SbAtomFile that feeds event-less successor and has larger order
            Assert(succEdgesWithoutEvent.size() == 1, "SbAtomFile must have one successor");
            {
                bool foundHigherPrecedentEdge = false;
                auto succWaveop = succEdgesWithoutEvent[0]->gToOp();
                for (auto precWaveEdge : succWaveop->gPrevWaveEdges()) {
                    auto precWaveop = precWaveEdge->gFromOp();
                    if (precWaveop == sbAtomFileWaveOp) {
                        continue;
                    }
                    if (dynamic_cast<wave::SbAtomFileWaveOp*>(precWaveop)  // must be SbAtomFile
                        && precWaveEdge->gEventId() != EventId_Invalid       // must have event
                        && sbAtomFileWaveOp->gOrder() < precWaveEdge->gFromOp()->gOrder()) // must be later
                    {
                        foundHigherPrecedentEdge = true;
                    }
                }
                Assert(foundHigherPrecedentEdge, "At least one SbAtomFile should have an event to its successor waveop");
            }
            break;
        default:
            Assert(false, "SbAtomFile ", sbAtomFileWaveOp->gName(),
                " must not have more than one successor. Number successors: MatMul=", succMatmulEdges.size(),
                ", Activation=", succActivationEdges.size(), ", Pool=", succPoolEdges.size());
            break;
        }

        if (succWaveEdgeEmb) {
            setEventId = succWaveEdgeEmb->gEventId();
            eventSetMode = succWaveEdgeEmb->gSetEventMode();
        }

    }

    //************************************************************************
    // Instruction(s)
    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomFileWaveOp->gIfmapCount();
    const kcc_int64 numBytesPerPart = sbAtomFileWaveOp->gLength();
    const kcc_int64 addressInPart   = sbAtomFileWaveOp->gAddressInPartition(0 /*offset in atom*/);
    const kcc_int64 stepSize = sbAtomFileWaveOp->gPartitionStepBytes();

    dramToStateBufInstr.nbytes                  = numBytesPerPart;

    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
        if (qParallelStreams()) {
            if (numPartitions-1 == partIdx) {
                // only the last reading sets event to enable subsequent instr
                dramToStateBufInstr.sync.set_event_id       = setEventId;
                dramToStateBufInstr.sync.set_event_mode     = events::eventSetMode2Int(eventSetMode);
            } else {
                dramToStateBufInstr.sync.set_event_id       = -1;
                dramToStateBufInstr.sync.set_event_mode     = eventSetMode2Int(events::EventSetMode::NoEvent);
            }
        }


        dramToStateBufInstr.src_address = npyFileDramOffset + sbAtomFileWaveOp->gOffsetInFile() + (partIdx * stepSize);
        dramToStateBufInstr.dst_address = stateBuf.gEntrySysAddress(partIdx, addressInPart);

        m_WaveCode.writeInstruction(dramToStateBufInstr);
    }
    //************************************************************************
}

}}

