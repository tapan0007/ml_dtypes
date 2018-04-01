#include "utils/inc/asserter.hpp"

#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisaldweights.hpp"
#include "compisa/inc/compisamatmul.hpp"

#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/barrierwaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodematmul.hpp"

namespace kcc {
namespace wavecode {


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

    compisa::LdWeightsInstr ldweightsInstr;

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

    ldweightsInstr.sync.wait_event_id    = 0;
    ldweightsInstr.sync.wait_event_mode  = events::eventWaitMode2Int(events::EventWaitMode::DontWait);
    ldweightsInstr.sync.set_event_id     = 0;
    ldweightsInstr.sync.set_event_mode   = events::eventSetMode2Int(events::EventSetMode::DontSet);

    //************************************************************************
    // incoming events
    //************************************************************************
    bool firstEmbEvt = true;
    if (qParallelStreams()) { // incoming events
        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (! prevWaveEdge->qNeedToImplementWait()) {
                continue;
            }
            if (! qLoadWeightsWaitsFor(prevWaveEdge)) {
                continue;
            }

            if (firstEmbEvt) {
                firstEmbEvt = false;
                ldweightsInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                ldweightsInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
            } else {
                writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
            }
        }
    }


    //************************************************************************
    // No outgoing events
    //************************************************************************

    //************************************************************************
    m_WaveCode.writeInstruction(ldweightsInstr);
}


bool
WaveCodeMatMul::qLoadWeightsWaitsFor(const wave::WaveEdge* prevEdge) const
{
    const auto prevWaveop = prevEdge->gFromOp();
    if (auto prevSbAtomLoadWaveop = dynamic_cast<const wave::SbAtomLoadWaveOp*>(prevWaveop)) {
        if (prevSbAtomLoadWaveop->qContainWeights()) {
            return true;
        }
    }
    if (dynamic_cast<const wave::BarrierWaveOp*>(prevWaveop)) {
        return true;
    }
    return false;
}



void
WaveCodeMatMul::generateMatMul(wave::MatMulWaveOp* matmulWaveop)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    //const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = matmulWaveop->gEngineId();
    Assert(EngineId::PeArray == engineId, "Engine id for MatMul should be PeArray");

    compisa::MatMulInstr matmulInstr;
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

    matmulInstr.sync.wait_event_id    = 0;
    matmulInstr.sync.wait_event_mode  = events::eventWaitMode2Int(events::EventWaitMode::DontWait);
    matmulInstr.sync.set_event_id    = 0;
    matmulInstr.sync.set_event_mode  = events::eventSetMode2Int(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        bool firstEmb = true;

        // Inspect incoming edges/events
        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (! prevWaveEdge->qNeedToImplementWait()) {
                continue;
            }
            if (qLoadWeightsWaitsFor(prevWaveEdge)) {
                continue;
            }

            if (firstEmb) {
                firstEmb = false;
                matmulInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                matmulInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
            } else {
                writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
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


