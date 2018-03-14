#include "shared/inc/tpb_isa_activate.hpp"


#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodeactivation.hpp"

namespace kcc {
namespace wavecode {

#define ASSERT_HAS_EVENT(edge, from, to) Assert((edge)->gEventId() != EventId_Invalid, "WaveEdge from waveop ", \
            (from)->gName(), " to waveop ", (to)->gName(), " has no event")

WaveCodeActivation::WaveCodeActivation(WaveCode* waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeActivation::generate(wave::WaveOp* waveOp)
{
    auto activationWaveOp = dynamic_cast<wave::ActivationWaveOp*>(waveOp);
    assert(activationWaveOp);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());
    const EngineId engineId = activationWaveOp->gEngineId();

    ACTIVATION activationInstr;

    activationInstr.activation_func     = activationWaveOp->gSimActivationFunc();
    activationInstr.in_dtype            = activationWaveOp->gInDtype().gSimTypeId();
    activationInstr.bias_dtype          = activationWaveOp->gBiasDtype().gSimTypeId();
    activationInstr.out_dtype           = activationWaveOp->gOutDtype().gSimTypeId();

    // TODO: for now Activation reads from 0 elem in bank.
    activationInstr.src_start_addr      = psumBuf.gEntryTpbAddress(activationWaveOp->gSrcPsumBankId(), 0, activationWaveOp->gInDtype());

    activationInstr.src_x_step          = activationWaveOp->gSrcXStep();
    activationInstr.src_y_step          = activationWaveOp->gSrcYStep();
    // activationInstr.src_z_step          = activationWaveOp->gSrcZStep(); // when available in the new ISA
    activationInstr.src_x_num           = activationWaveOp->gSrcXNum();
    activationInstr.src_y_num           = activationWaveOp->gSrcYNum();
    // activationInstr.src_z_num           = activationWaveOp->gSrcZNum(); // when available in the new ISA

    if (activationWaveOp->qDstIsPsum()) {
        activationInstr.dst_start_addr  = psumBuf.gEntryTpbAddress(activationWaveOp->gDstPsumBankId(),
                                                                  0, /* bank offset 0 */
                                                                  activationWaveOp->gOutDtype());
    } else {
        activationInstr.dst_start_addr  = stateBuf.gEntryTpbAddress(0, /* row 0 */
                                                activationWaveOp->gDstSbAtomId() * activationWaveOp->gWaveAtomSize()
                                                    + activationWaveOp->gDstSbOffsetInAtom());
    }
    activationInstr.dst_x_step      = activationWaveOp->gDstXStep();
    activationInstr.dst_y_step      = activationWaveOp->gDstYStep();
    activationInstr.dst_z_step      = activationWaveOp->gDstZStep();
    activationInstr.dst_x_num       = activationWaveOp->gDstXNum();
    activationInstr.dst_y_num       = activationWaveOp->gDstYNum();
    activationInstr.dst_z_num       = activationWaveOp->gDstZNum();

    activationInstr.scale_value         = activationWaveOp->gScale();
    if (activationWaveOp->qBiasAddEn ()) {
        activationInstr.acc_addr        = stateBuf.gEntryTpbAddress(
                                            0,   //row 0 for now
                                            activationWaveOp->gBiasAtomId() * activationWaveOp->gWaveAtomSize()
                                                + activationWaveOp->gBiasOffsetInAtom());
    } else {
        activationInstr.acc_addr        = stateBuf.gAllZeroOffsetTpbAddress(activationWaveOp->gBiasDtype());
    }
    activationInstr.num_partitions      = activationWaveOp->gNumPartitions();

    //************************************************************************
    { // incoming events
        std::vector<const wave::WaveEdge*> prevIfmapEdges;
        std::vector<const wave::WaveEdge*> prevMatmulEdges;
        std::vector<const wave::WaveEdge*> prevPoolEdges;

        // Inspect incoming edges/events
        for (auto prevWaveEdge : activationWaveOp->gPrevWaveEdges()) {
            if (prevWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto prevWaveop = prevWaveEdge->gFromOp();
            if (prevWaveop->gEngineId() == engineId) {
                continue;
            }
            if (auto prevSbAtomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevSbAtomLoadWaveop, activationWaveOp);
                Assert(!prevSbAtomLoadWaveop->qContainWeights(), "SbAtomLoad ", prevSbAtomLoadWaveop->gName(),
                       " preceeding Activation ", activationWaveOp->gName(), " cannot contain weights");
                prevIfmapEdges.push_back(prevWaveEdge);
                continue;
            }
            if (auto prevPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevPoolWaveop, activationWaveOp);
                prevPoolEdges.push_back(prevWaveEdge);
                continue;
            }
            if (auto prevMatmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevMatmulWaveop, activationWaveOp);
                prevMatmulEdges.push_back(prevWaveEdge);
                continue;
            }
            Assert(false, "Activation waveop: predecessor waveop ", prevWaveop->gName(), " has wrong type ", prevWaveop->gTypeStr());
        }

        bool firstEmb = true;
        for (auto prevWaveEdge : prevIfmapEdges) {
            if (firstEmb) {
                activationInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                activationInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode->writeInstruction(waitInstr, EngineId::Pooling);
            }
        }
        for (auto prevWaveEdge : prevPoolEdges) {
            if (firstEmb) {
                activationInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                activationInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode->writeInstruction(waitInstr, EngineId::Pooling);
            }
        }
        for (auto prevWaveEdge : prevMatmulEdges) {
            if (firstEmb) {
                activationInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                activationInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode->writeInstruction(waitInstr, EngineId::Pooling);
            }
        }
    }


    //************************************************************************
    { // Outgoing events
        std::vector<const wave::WaveEdge*> succIfmapEdges;
        std::vector<const wave::WaveEdge*> succMatmulEdges;
        std::vector<const wave::WaveEdge*> succPoolEdges;

        for (auto succWaveEdge : activationWaveOp->gSuccWaveEdges()) {
            if (succWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto succWaveop = succWaveEdge->gToOp();
            if (succWaveop->gEngineId() == engineId) {
                continue;
            }

            if (auto succSbAtomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, activationWaveOp, succSbAtomSaveWaveop);
                succIfmapEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succMatmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, activationWaveOp, succMatmulWaveop);
                succMatmulEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, activationWaveOp, succPoolWaveop);
                succPoolEdges.push_back(succWaveEdge);
                continue;
            }
            Assert(false, "Activation waveop: successor waveop ", succWaveop->gName(), " has wrong type ", succWaveop->gTypeStr());
        }

        bool firstEmb = true;
        for (auto succWaveEdge : succIfmapEdges) {
            WRITE writeInstr;
            writeInstr.dst_address  = m_WaveCode->calculateEventAddress(EngineId::DmaEng, succWaveEdge->gEventId());
            writeInstr.data         = ~(0UL);  // writing is for remote event-set. All 1's ensure that bit/byte endianess does not matter.
            writeInstr.nbytes       = 1;

            m_WaveCode->writeInstruction(writeInstr, EngineId::DmaEng);
        }
        for (auto succWaveEdge : succMatmulEdges) {
            if (firstEmb) {
                activationInstr.sync.set_event_id       = succWaveEdge->gEventId();
                activationInstr.sync.set_event_mode     = events::eventSetMode2Int(succWaveEdge->gSetEventMode());
                firstEmb = false;
            } else {
                SET setEventInstr;
                setEventInstr.event_id          = succWaveEdge->gEventId();
                m_WaveCode->writeInstruction(setEventInstr, EngineId::PeArray);
            }
        }
        for (auto succWaveEdge : succPoolEdges) {
            if (firstEmb) {
                activationInstr.sync.set_event_id       = succWaveEdge->gEventId();
                activationInstr.sync.set_event_mode     = events::eventSetMode2Int(succWaveEdge->gSetEventMode());
                firstEmb = false;
            } else {
                SET setEventInstr;
                setEventInstr.event_id          = succWaveEdge->gEventId();
                m_WaveCode->writeInstruction(setEventInstr, EngineId::Activation);
            }
        }
    }

    //************************************************************************
    // write instruction
    m_WaveCode->writeInstruction(activationInstr);
}


}}


