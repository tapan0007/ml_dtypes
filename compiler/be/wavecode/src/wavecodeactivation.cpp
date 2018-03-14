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
WaveCodeActivation::generate(wave::WaveOp* waveop)
{
    auto activationWaveop = dynamic_cast<wave::ActivationWaveOp*>(waveop);
    assert(activationWaveop);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());
    const EngineId engineId = activationWaveop->gEngineId();

    ACTIVATION activationInstr;

    activationInstr.activation_func     = activationWaveop->gSimActivationFunc();
    activationInstr.in_dtype            = activationWaveop->gInDtype().gSimTypeId();
    activationInstr.bias_dtype          = activationWaveop->gBiasDtype().gSimTypeId();
    activationInstr.out_dtype           = activationWaveop->gOutDtype().gSimTypeId();

    // TODO: for now Activation reads from 0 elem in bank.
    activationInstr.src_start_addr      = psumBuf.gEntryTpbAddress(activationWaveop->gSrcPsumBankId(), 0, activationWaveop->gInDtype());

    activationInstr.src_x_step          = activationWaveop->gSrcXStep();
    activationInstr.src_y_step          = activationWaveop->gSrcYStep();
    // activationInstr.src_z_step          = activationWaveop->gSrcZStep(); // when available in the new ISA
    activationInstr.src_x_num           = activationWaveop->gSrcXNum();
    activationInstr.src_y_num           = activationWaveop->gSrcYNum();
    // activationInstr.src_z_num           = activationWaveop->gSrcZNum(); // when available in the new ISA

    if (activationWaveop->qDstIsPsum()) {
        activationInstr.dst_start_addr  = psumBuf.gEntryTpbAddress(activationWaveop->gDstPsumBankId(),
                                                                  0, /* bank offset 0 */
                                                                  activationWaveop->gOutDtype());
    } else {
        activationInstr.dst_start_addr  = stateBuf.gEntryTpbAddress(0, /* row 0 */
                                                activationWaveop->gDstSbAtomId() * activationWaveop->gWaveAtomSize()
                                                    + activationWaveop->gDstSbOffsetInAtom());
    }
    activationInstr.dst_x_step      = activationWaveop->gDstXStep();
    activationInstr.dst_y_step      = activationWaveop->gDstYStep();
    activationInstr.dst_z_step      = activationWaveop->gDstZStep();
    activationInstr.dst_x_num       = activationWaveop->gDstXNum();
    activationInstr.dst_y_num       = activationWaveop->gDstYNum();
    activationInstr.dst_z_num       = activationWaveop->gDstZNum();

    activationInstr.scale_value         = activationWaveop->gScale();
    if (activationWaveop->qBiasAddEn ()) {
        activationInstr.acc_addr        = stateBuf.gEntryTpbAddress(
                                            0,   //row 0 for now
                                            activationWaveop->gBiasAtomId() * activationWaveop->gWaveAtomSize()
                                                + activationWaveop->gBiasOffsetInAtom());
    } else {
        activationInstr.acc_addr        = stateBuf.gAllZeroOffsetTpbAddress(activationWaveop->gBiasDtype());
    }
    activationInstr.num_partitions      = activationWaveop->gNumPartitions();

    //************************************************************************
    { // incoming events
        std::vector<const wave::WaveEdge*> prevIfmapEdges;
        std::vector<const wave::WaveEdge*> prevMatmulEdges;
        std::vector<const wave::WaveEdge*> prevPoolEdges;

        // Inspect incoming edges/events
        for (auto prevWaveEdge : activationWaveop->gPrevWaveEdges()) {
            if (prevWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto prevWaveop = prevWaveEdge->gFromOp();
            if (prevWaveop->gEngineId() == engineId) {
                continue;
            }
            if (auto prevSbAtomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevSbAtomLoadWaveop, activationWaveop);
                Assert(!prevSbAtomLoadWaveop->qContainWeights(), "SbAtomLoad ", prevWaveop->gName(),
                       " preceeding Activation ", activationWaveop->gName(), " cannot contain weights");
                prevIfmapEdges.push_back(prevWaveEdge);
                continue;
            }
            if (auto prevPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevPoolWaveop, activationWaveop);
                prevPoolEdges.push_back(prevWaveEdge);
                continue;
            }
            if (auto prevMatmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevMatmulWaveop, activationWaveop);
                prevMatmulEdges.push_back(prevWaveEdge);
                continue;
            }
            Assert(false, "Activation waveop ", activationWaveop->gName(), ": predecessor waveop ", prevWaveop->gName(),
                   " has wrong type ", prevWaveop->gTypeStr());
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
        std::vector<const wave::WaveEdge*> succOfmapEdges;
        std::vector<const wave::WaveEdge*> succMatmulEdges;
        std::vector<const wave::WaveEdge*> succPoolEdges;

        for (auto succWaveEdge : activationWaveop->gSuccWaveEdges()) {
            if (succWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto succWaveop = succWaveEdge->gToOp();
            if (succWaveop->gEngineId() == engineId) {
                continue;
            }

            if (auto succSbAtomSaveWaveOp = dynamic_cast<wave::SbAtomSaveWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, activationWaveop, succSbAtomSaveWaveOp);
                succOfmapEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succMatmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, activationWaveop, succMatmulWaveop);
                succMatmulEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, activationWaveop, succPoolWaveop);
                succPoolEdges.push_back(succWaveEdge);
                continue;
            }
            Assert(false, "Activation waveop ", activationWaveop->gName(), ": successor waveop ", succWaveop->gName(),
                   " has wrong type ", succWaveop->gTypeStr());
        }

        bool firstEmb = true;
        for (auto succWaveEdge : succOfmapEdges) {
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


