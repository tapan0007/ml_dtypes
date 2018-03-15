#include "shared/inc/tpb_isa_pool.hpp"

#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "layers/inc/poollayer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodepool.hpp"

namespace kcc {
namespace wavecode {

#define ASSERT_HAS_EVENT(edge, from, to) Assert((edge)->gEventId() != EventId_Invalid, "WaveEdge from waveop ", \
            (from)->gName(), " to waveop ", (to)->gName(), " has no event")

WaveCodePool::WaveCodePool(WaveCode* waveCode)
    : WaveCodeWaveOp(waveCode)
{}


void
WaveCodePool::generate(wave::WaveOp* waveOp)
{
    auto poolWaveop = dynamic_cast<wave::PoolWaveOp*>(waveOp);
    assert(poolWaveop);
    const arch::Arch& arch(arch::Arch::gArch());
    const auto& psumBuf(arch.gPsumBuffer());
    const auto& stateBuf(arch.gStateBuffer());

    const EngineId engineId = poolWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for Pool should be Pooling");

    POOL poolInstr;

    /* pool args */
    switch (poolWaveop->gPoolFunc()) {
    case PoolType::Max:
        poolInstr.pool_func = POOLFUNC::MAX_POOL;
        break;
    case PoolType::Avg:
        poolInstr.pool_func = POOLFUNC::AVG_POOL;
        break;
    default:
        assert(false && "Bad PoolType in PoolWaveOp");
        break;
    }

    poolInstr.in_dtype          = poolWaveop->gInDtype().gSimTypeId();
    poolInstr.out_dtype         = poolWaveop->gOutDtype().gSimTypeId();
    if (poolWaveop->qSrcIsPsum()) {
        poolInstr.src_start_addr = psumBuf.gEntryTpbAddress(
                                            poolWaveop->gSrcPsumBankId(),
                                            poolWaveop->gSrcPsumBankOffset(),
                                            poolWaveop->gInDtype());
    } else { // State buffer
        poolInstr.src_start_addr = stateBuf.gEntryTpbAddress(0, /*row 0 for now*/
                                            poolWaveop->gSrcSbAtomId() * poolWaveop->gWaveAtomSize()
                                                + poolWaveop->gSrcSbOffsetInAtom());
    }

    poolInstr.src_x_step        = poolWaveop->gSrcXStep();
    poolInstr.src_x_num         = poolWaveop->gSrcXNum();
    poolInstr.src_y_step        = poolWaveop->gSrcYStep();
    poolInstr.src_y_num         = poolWaveop->gSrcYNum();
    poolInstr.pool_frequency    = poolWaveop->gPoolFrequency();
    poolInstr.pool_scale        = static_cast<float>(1.0/poolWaveop->gPoolFrequency());
    /* strides */
    poolInstr.src_z_step        = poolWaveop->gSrcZStep();
    poolInstr.src_z_num         = poolWaveop->gSrcZNum();
    poolInstr.src_w_step        = poolWaveop->gSrcWStep();
    poolInstr.src_w_num         = poolWaveop->gSrcWNum();
    poolInstr.num_partitions    = poolWaveop->gNumPartitions();

    /* Pool  */
    poolInstr.dst_x_step        = poolWaveop->gDstXStep();
    poolInstr.dst_x_num         = poolWaveop->gDstXNum();
    poolInstr.dst_y_step        = poolWaveop->gDstYStep();
    poolInstr.dst_y_num         = poolWaveop->gDstYNum();
    poolInstr.dst_z_step        = poolWaveop->gDstZStep();
    poolInstr.dst_z_num         = poolWaveop->gDstZNum();
    // For now DST is always StateBuf
    poolInstr.dst_start_addr    = stateBuf.gEntryTpbAddress(0, /*row 0 for now*/
                                            poolWaveop->gDstSbAtomId() * poolWaveop->gWaveAtomSize()
                                                + poolWaveop->gDstSbOffsetInAtom());

    //************************************************************************
    { // incoming events
        std::vector<const wave::WaveEdge*> prevIfmapEdges;
        std::vector<const wave::WaveEdge*> prevMatmulEdges;
        std::vector<const wave::WaveEdge*> prevActivationEdges;

        // Inspect incoming edges/events
        for (auto prevWaveEdge : poolWaveop->gPrevWaveEdges()) {
            if (prevWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto prevWaveop = prevWaveEdge->gFromOp();
            if (prevWaveop->gEngineId() == engineId) {
                continue;
            }
            if (auto prevSbAtomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevSbAtomLoadWaveop, poolWaveop);
                Assert(!prevSbAtomLoadWaveop->qContainWeights(), "SbAtomLoad ", prevSbAtomLoadWaveop->gName(),
                       " preceeding Pool ", poolWaveop->gName(), " cannot contain weights");
                prevIfmapEdges.push_back(prevWaveEdge);
                continue;
            }
            if (auto prevActivationWaveop = dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevActivationWaveop, poolWaveop);
                prevActivationEdges.push_back(prevWaveEdge);
                continue;
            }
            if (auto prevMatmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevMatmulWaveop, poolWaveop);
                prevMatmulEdges.push_back(prevWaveEdge);
                continue;
            }
            Assert(false, "Pool waveop ", poolWaveop->gName(), ": predecessor waveop ", prevWaveop->gName(),
                   " has wrong type ", prevWaveop->gTypeStr());
        }

        bool firstEmb = true;
        for (auto prevWaveEdge : prevIfmapEdges) {
            if (firstEmb) {
                poolInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                poolInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode->writeInstruction(waitInstr, engineId);
            }
        }
        for (auto prevWaveEdge : prevActivationEdges) {
            if (firstEmb) {
                poolInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                poolInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode->writeInstruction(waitInstr, engineId);
            }
        }
        for (auto prevWaveEdge : prevMatmulEdges) {
            if (firstEmb) {
                poolInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                poolInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode->writeInstruction(waitInstr, engineId);
            }
        }
    } // end incoming events


    //************************************************************************
    { // Outgoing events
        std::vector<const wave::WaveEdge*> succOfmapEdges;
        std::vector<const wave::WaveEdge*> succMatmulEdges;
        std::vector<const wave::WaveEdge*> succActivationEdges;

        for (auto succWaveEdge : poolWaveop->gSuccWaveEdges()) {
            if (succWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto succWaveop = succWaveEdge->gToOp();
            if (succWaveop->gEngineId() == engineId) {
                continue;
            }

            if (auto succSbAtomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, poolWaveop, succSbAtomSaveWaveop);
                succOfmapEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succMatmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, poolWaveop, succMatmulWaveop);
                succMatmulEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succActivationWaveop = dynamic_cast<wave::ActivationWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, poolWaveop, succActivationWaveop);
                succActivationEdges.push_back(succWaveEdge);
                continue;
            }
            Assert(false, "Pool waveop ", poolWaveop->gName(), ": successor waveop ", succWaveop->gName(),
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
        if (succWaveEdgeEmb) {
            poolInstr.sync.set_event_id    = succWaveEdgeEmb->gEventId();
            poolInstr.sync.set_event_mode  = events::eventSetMode2Int(succWaveEdgeEmb->gSetEventMode());
        }

        //************************************************************************
        // write instruction
        m_WaveCode->writeInstruction(poolInstr);
        //************************************************************************


        //************************************************************************
        // Remaining edges --> signal through SET_EVENT or through WRITE
        //************************************************************************
        for (kcc_uint32 matmulIdx = matmulStart; matmulIdx < succMatmulEdges.size(); ++matmulIdx) {
            SET setEventInstr;
            auto succWaveEdge               = succMatmulEdges[matmulIdx];
            setEventInstr.event_id          = succWaveEdge->gEventId();
            m_WaveCode->writeInstruction(setEventInstr, engineId);
        }
        for (kcc_uint32 activationIdx = activationStart; activationIdx < succActivationEdges.size(); ++activationIdx) {
            SET setEventInstr;
            auto succWaveEdge               = succActivationEdges[activationIdx];
            setEventInstr.event_id          = succWaveEdge->gEventId();
            m_WaveCode->writeInstruction(setEventInstr, engineId);
        }
        for (auto succWaveEdge : succOfmapEdges) {
            WRITE writeInstr; // writing is for remote event-set
            writeInstr.dst_address          = m_WaveCode->calculateEventAddress(EngineId::DmaEng, succWaveEdge->gEventId());
            writeInstr.data                 = ~(0UL);  // All 1's => bit/byte endianess does not matter.
            writeInstr.nbytes               = 1;

            m_WaveCode->writeInstruction(writeInstr, engineId);
        }
    }
}

}}


