#include "shared/inc/tpb_isa_wait.hpp"
#include "shared/inc/tpb_isa_ldweights.hpp"
#include "shared/inc/tpb_isa_matmul.hpp"

#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodematmul.hpp"

namespace kcc {
namespace wavecode {

#define ASSERT_HAS_EVENT(edge, from, to) Assert((edge)->gEventId() != EventId_Invalid, "WaveEdge from waveop ", \
            (from)->gName(), " to waveop ", (to)->gName(), " has no event")

WaveCodeMatMul::WaveCodeMatMul(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeMatMul::generate(wave::WaveOp* waveOp)
{
    auto matmulWaveOp = dynamic_cast<wave::MatMulWaveOp*>(waveOp);
    assert(matmulWaveOp);

    generateLoadWeights(matmulWaveOp);
    generateMatMul(matmulWaveOp);
}



void
WaveCodeMatMul::generateLoadWeights(wave::MatMulWaveOp* matmulWaveop)
{
    assert(matmulWaveop->verify());
    if (matmulWaveop->gWeightsSbAddress() < 0) {
        return; // this MatMul reuses weights
    }
    //const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    //const wave::MatMulWaveOp::WaveId& waveId(matmulWaveop->gWaveId());

    LDWEIGHTS ldweightsInstr;

    //TPB_CMD_HEADER  hdr;
    const utils::DataType& dtype(matmulWaveop->gInDtype());
    ldweightsInstr.dtype                 = dtype.gSimTypeId();
    //uint8_t         perf_opt = OPT_NONE;
    //uint8_t         dquant_table_idx  = 0;
    //uint8_t         dquant_in_dsize   = 0;
    //uint8_t         dquant_out_dtype  = INVALID_ARBPRECTYPE;
    //uint8_t         dquant_enable  = 0;
    ///* subtract this from each ldweights on way into PE Array */
    //union {
    //    uint8_t     zero_point_uint8[2];
    //    uint16_t    zero_point_uint16   = 0;
    //} TONGA_PACKED;
    const kcc_int64 addressInSbPart     = matmulWaveop->gWeightsSbAddress();

    ldweightsInstr.start_addr            = addressInSbPart + (matmulWaveop->gOfmapCount() - 1) * dtype.gSizeInBytes();

    ldweightsInstr.x_step                = -1; // last column goes first, so decrement
    ldweightsInstr.x_num                 = matmulWaveop->gOfmapCount();
    ldweightsInstr.num_row_partitions    = matmulWaveop->gIfmapCount();

    ldweightsInstr.sync.wait_event_id    = -1;
    ldweightsInstr.sync.wait_event_mode  = events::eventWaitMode2Int(events::EventWaitMode::NoEvent);
    ldweightsInstr.sync.set_event_id    = -1;
    ldweightsInstr.sync.set_event_mode  = events::eventSetMode2Int(events::EventSetMode::NoEvent);

    //************************************************************************
    // incoming events
    //************************************************************************
    if (qParallelStreams()) { // incoming events
        std::vector<const wave::WaveEdge*> prevWeightEdges;
        std::vector<const wave::WaveEdge*> prevIfmapEdges;
        std::vector<const wave::WaveEdge*> prevPoolEdges;
        std::vector<const wave::WaveEdge*> prevActivationEdges;

        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (prevWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto prevWaveop = prevWaveEdge->gFromOp();
            if (auto prevSbAtomLoadWaveop = dynamic_cast<wave::SbAtomFileWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevSbAtomLoadWaveop, matmulWaveop);
                if (prevSbAtomLoadWaveop->qContainWeights()) {
                    prevWeightEdges.push_back(prevWaveEdge);
                } else {
                    prevIfmapEdges.push_back(prevWaveEdge);
                }
                continue;
            }
            if (auto prevPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevPoolWaveop, matmulWaveop);
                prevPoolEdges.push_back(prevWaveEdge);
                continue;
            }
            if (auto prevActivationWaveop = dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevActivationWaveop, matmulWaveop);
                prevActivationEdges.push_back(prevWaveEdge);
                continue;
            }
            Assert(false, "Matmul waveop ", matmulWaveop->gName(), ": waveop ", prevWaveop->gName(),
                " has wrong type ", prevWaveop->gTypeStr());
        }

        Assert(prevWeightEdges.size() <= 1, "Matmul waveop ", matmulWaveop->gName(), " can have only one weight predecessor");
        if (prevWeightEdges.size() > 0) {
            auto prevWaveEdge = prevWeightEdges[0];
            //auto prevWaveop = prevWaveEdge->gFromOp();
            ldweightsInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
            ldweightsInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
        }
    }

    //************************************************************************
    // No outgoing events
    //************************************************************************

    //************************************************************************
    // write instruction
    m_WaveCode.writeInstruction(ldweightsInstr);
    //************************************************************************
}



void
WaveCodeMatMul::generateMatMul(wave::MatMulWaveOp* matmulWaveop)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    //const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = matmulWaveop->gEngineId();
    Assert(EngineId::PeArray == engineId, "Engine id for MatMul should be PeArray");

    MATMUL matmulInstr;
    matmulInstr.dtype                   = matmulWaveop->gInDtype().gSimTypeId();
    matmulInstr.num_row_partitions      = matmulWaveop->gNumRowPartitions();
    matmulInstr.num_column_partitions   = matmulWaveop->gNumColumnPartitions();

    matmulInstr.fmap_start_addr         = matmulWaveop->gIfmapsSbAddress();
    matmulInstr.fmap_x_num              = matmulWaveop->gFmapXNum();
    matmulInstr.fmap_x_step             = matmulWaveop->gFmapXStep();
    matmulInstr.fmap_y_num              = matmulWaveop->gFmapYNum();
    matmulInstr.fmap_y_step             = matmulWaveop->gFmapYStep();
    matmulInstr.fmap_z_num              = matmulWaveop->gFmapZNum();
    matmulInstr.fmap_z_step             = 1;

    matmulInstr.psum_start_addr         = psumBuf.gEntryTpbAddress(
                                                    matmulWaveop->gPsumBankId(),
                                                    matmulWaveop->gPsumBankOffset(),
                                                    matmulWaveop->gOutDtype());
    matmulInstr.psum_x_num              = matmulWaveop->gPsumXNum();
    matmulInstr.psum_x_step             = matmulWaveop->gPsumXStep();
    matmulInstr.psum_y_num              = matmulWaveop->gPsumYNum();
    matmulInstr.psum_y_step             = matmulWaveop->gPsumYStep();

    matmulInstr.start_tensor_calc       = matmulWaveop->qStartTensorCalc();
    matmulInstr.stop_tensor_calc        = matmulWaveop->qStopTensorCalc();

    matmulInstr.sync.wait_event_id    = -1;
    matmulInstr.sync.wait_event_mode  = events::eventWaitMode2Int(events::EventWaitMode::NoEvent);
    matmulInstr.sync.set_event_id    = -1;
    matmulInstr.sync.set_event_mode  = events::eventSetMode2Int(events::EventSetMode::NoEvent);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        // Right now all non-weight prev edges are treated equally, but in
        // the future we may change it, so keep separate arrays.
        std::vector<const wave::WaveEdge*> prevWeightEdges;
        std::vector<const wave::WaveEdge*> prevIfmapEdges;
        std::vector<const wave::WaveEdge*> prevPoolEdges;
        std::vector<const wave::WaveEdge*> prevActivationEdges;

        // Inspect incoming edges/events
        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (prevWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto prevWaveop = prevWaveEdge->gFromOp();
            if (prevWaveop->gEngineId() == engineId) {
                continue;
            }
            if (auto prevSbAtomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevSbAtomLoadWaveop, matmulWaveop);
                if (prevSbAtomLoadWaveop->qContainWeights()) {
                    prevWeightEdges.push_back(prevWaveEdge);
                } else {
                    prevIfmapEdges.push_back(prevWaveEdge);
                }
                continue;
            }
            if (auto prevPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevPoolWaveop, matmulWaveop);
                prevPoolEdges.push_back(prevWaveEdge);
                continue;
            }
            if (auto prevActivationWaveop = dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevActivationWaveop, matmulWaveop);
                prevActivationEdges.push_back(prevWaveEdge);
                continue;
            }
            Assert(false, "Matmul waveop ", matmulWaveop->gName(), ": predecessor waveop ", prevWaveop->gName(),
                   " has wrong type ", prevWaveop->gTypeStr());
        }

        bool firstEmb = true;
        for (auto prevWaveEdge : prevIfmapEdges) {
            if (firstEmb) {
                matmulInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                matmulInstr.sync.wait_event_mode    = events::eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode.writeInstruction(waitInstr, engineId);
            }
        }
        for (auto prevWaveEdge : prevPoolEdges) {
            if (firstEmb) {
                matmulInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                matmulInstr.sync.wait_event_mode    = events::eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode.writeInstruction(waitInstr, engineId);
            }
        }
        for (auto prevWaveEdge : prevActivationEdges) {
            if (firstEmb) {
                matmulInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                matmulInstr.sync.wait_event_mode    = events::eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode.writeInstruction(waitInstr, engineId);
            }
        }
    } // end incoming events



    std::vector<const wave::WaveEdge*> succOfmapEdges;
    std::vector<const wave::WaveEdge*> succPoolEdges;
    std::vector<const wave::WaveEdge*> succActivationEdges;
    kcc_uint32 poolStart = 0;
    kcc_uint32 activationStart = 0;
    kcc_uint32 succOfmapStart = 0;

    //************************************************************************
    if (qParallelStreams()) { // Outgoing events
        // Right now all succ edges are treated equally, but in the future
        // we may change it, so keep separate arrays.

        for (auto succWaveEdge : matmulWaveop->gSuccWaveEdges()) {
            if (succWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto succWaveop = succWaveEdge->gToOp();
            if (succWaveop->gEngineId() == engineId) {
                continue;
            }
            if (auto succSbAtomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, matmulWaveop, succSbAtomSaveWaveop);
                succOfmapEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, matmulWaveop, succPoolWaveop);
                succPoolEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succActivationWaveop = dynamic_cast<wave::ActivationWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, matmulWaveop, succActivationWaveop);
                succActivationEdges.push_back(succWaveEdge);
                continue;
            }
            Assert(false, "Matmul waveop: successor waveop ", succWaveop->gName(), " has wrong type ", succWaveop->gTypeStr());
        }

        //************************************************************************
        // Find one embedded set-event edge, if any
        //************************************************************************
        const wave::WaveEdge* succWaveEdgeEmb  = nullptr;
        if (!succWaveEdgeEmb && succPoolEdges.size() > 0) {
            succWaveEdgeEmb = succPoolEdges[poolStart++];
        }
        if (!succWaveEdgeEmb && succActivationEdges.size() > 0) {
            succWaveEdgeEmb = succActivationEdges[activationStart++];
        }
        if (!succWaveEdgeEmb && succOfmapEdges.size() > 0) {
            succWaveEdgeEmb = succOfmapEdges[succOfmapStart++];
        }
        if (succWaveEdgeEmb) {
            matmulInstr.sync.set_event_id   = succWaveEdgeEmb->gEventId();
            matmulInstr.sync.set_event_mode = events::eventSetMode2Int(succWaveEdgeEmb->gSetEventMode());
        }
    }


    //************************************************************************
    // write instruction
    m_WaveCode.writeInstruction(matmulInstr);
    //************************************************************************



    if (qParallelStreams()) { // Outgoing events
        //************************************************************************
        // Remaining edges --> signal through SET_EVENT
        //************************************************************************
        for (kcc_uint32 poolIdx = poolStart; poolIdx < succPoolEdges.size(); ++poolIdx) {
            SET setEventInstr;
            auto succWaveEdge               = succPoolEdges[poolIdx];
            setEventInstr.event_id          = succWaveEdge->gEventId();
            m_WaveCode.writeInstruction(setEventInstr, engineId);
        }
        for (kcc_uint32 activationIdx = activationStart; activationIdx < succActivationEdges.size(); ++activationIdx) {
            SET setEventInstr;
            auto succWaveEdge               = succActivationEdges[activationIdx];
            setEventInstr.event_id          = succWaveEdge->gEventId();
            m_WaveCode.writeInstruction(setEventInstr, engineId);
        }
        for (kcc_uint32 succOfmapIdx = succOfmapStart; succOfmapIdx < succOfmapEdges.size(); ++succOfmapIdx) {
            SET setEventInstr;
            auto succWaveEdge               = succOfmapEdges[succOfmapIdx];
            setEventInstr.event_id          = succWaveEdge->gEventId();
            m_WaveCode.writeInstruction(setEventInstr, engineId);
        }
    }
}


}}


