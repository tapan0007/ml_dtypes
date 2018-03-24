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
#include "wave/inc/sbatomloadwaveop.hpp"
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
    const EngineId engineId = matmulWaveop->gEngineId();
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
    ldweightsInstr.sync.set_event_id     = -1;
    ldweightsInstr.sync.set_event_mode   = events::eventSetMode2Int(events::EventSetMode::NoEvent);

    //************************************************************************
    // incoming events
    //************************************************************************
    bool firstEmbEvt = true;
    if (qParallelStreams()) { // incoming events
        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (! prevWaveEdge->qNeedToImplementWait()) {
                continue;
            }
            const auto prevWaveop = prevWaveEdge->gFromOp();

            if (auto prevSbAtomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
                if (prevSbAtomLoadWaveop->qContainWeights()) {
                    if (firstEmbEvt) {
                        firstEmbEvt = false;
                        ldweightsInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                        ldweightsInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                    } else {
                        WAIT waitInstr;
                        waitInstr.event_id  = prevWaveEdge->gEventId();
                        m_WaveCode.writeInstruction(waitInstr, engineId);
                    }
                }
                continue;
            }
        }
    }

    //************************************************************************
    // No outgoing events
    //************************************************************************

    //************************************************************************
    m_WaveCode.writeInstruction(ldweightsInstr);
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
        bool firstEmb = true;

        // Inspect incoming edges/events
        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (! prevWaveEdge->qNeedToImplementWait()) {
                continue;
            }
            const auto prevWaveop = prevWaveEdge->gFromOp();

            if (auto prevSbAtomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
                if (! prevSbAtomLoadWaveop->qContainWeights()) { // Load Ifmap
                    if (firstEmb) {
                        firstEmb = false;
                        matmulInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                        matmulInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                    } else {
                        WAIT waitInstr;
                        waitInstr.event_id  = prevWaveEdge->gEventId();
                        m_WaveCode.writeInstruction(waitInstr, engineId);
                    }
                }
                continue;
            } else {  // Save or non-sb instructions
                if (firstEmb) {
                    firstEmb = false;
                    matmulInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                    matmulInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                } else {
                    WAIT waitInstr;
                    waitInstr.event_id  = prevWaveEdge->gEventId();
                    m_WaveCode.writeInstruction(waitInstr, engineId);
                }
                continue;
            }
        }
    } // end incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(matmulWaveop, matmulInstr);
    }
    if (! instructionWritten) {
        m_WaveCode.writeInstruction(matmulInstr);
    }
}


}}


