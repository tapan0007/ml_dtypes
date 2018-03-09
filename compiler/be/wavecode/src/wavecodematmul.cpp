#include "shared/inc/tpb_isa_ldweights.hpp"
#include "shared/inc/tpb_isa_matmul.hpp"

#include "utils/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodematmul.hpp"

namespace kcc {
namespace wavecode {

WaveCodeMatMul::WaveCodeMatMul(WaveCode* waveCode)
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
WaveCodeMatMul::generateMatMul(wave::MatMulWaveOp* matmulWaveOp)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    //const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    MATMUL matmulInstr;
    matmulInstr.dtype                   = matmulWaveOp->gInDtype().gSimTypeId();
    matmulInstr.num_row_partitions      = matmulWaveOp->gNumRowPartitions();
    matmulInstr.num_column_partitions   = matmulWaveOp->gNumColumnPartitions();

    matmulInstr.fmap_start_addr         = matmulWaveOp->gIfmapsAtomId() * wave::MatMulWaveOp::AtomSize +
                                          matmulWaveOp->gIfmapsOffsetInAtom();
    matmulInstr.fmap_x_num              = matmulWaveOp->gFmapXNum();
    matmulInstr.fmap_x_step             = matmulWaveOp->gFmapXStep();
    matmulInstr.fmap_y_num              = matmulWaveOp->gFmapYNum();
    matmulInstr.fmap_y_step             = matmulWaveOp->gFmapYStep();
    matmulInstr.fmap_z_num              = matmulWaveOp->gFmapZNum();
    matmulInstr.fmap_z_step             = matmulWaveOp->gFmapZStepAtoms() * matmulWaveOp->gIfmapsAtomSize();

    matmulInstr.psum_start_addr         = psumBuf.gEntryTpbAddress(
                                                    matmulWaveOp->gPsumBankId(),
                                                    matmulWaveOp->gPsumBankOffset(),
                                                    matmulWaveOp->gOutDtype());
    matmulInstr.psum_x_num              = matmulWaveOp->gPsumXNum();
    matmulInstr.psum_x_step             = matmulWaveOp->gPsumXStep();
    matmulInstr.psum_y_num              = matmulWaveOp->gPsumYNum();
    matmulInstr.psum_y_step             = matmulWaveOp->gPsumYStep();

    matmulInstr.start_tensor_calc       = matmulWaveOp->qStartTensorCalc();
    matmulInstr.stop_tensor_calc        = matmulWaveOp->qStopTensorCalc();

    //setup_sync(matmulInstr.sync, -1, SET_EVENT_ON_END_WR_DST);
    matmulInstr.sync.wait_event_id      = matmulWaveOp->gWaitEventId();
    matmulInstr.sync.wait_event_mode    = eventWaitMode2Int(matmulWaveOp->gWaitEventMode());
    matmulInstr.sync.set_event_id       = matmulWaveOp->gSetEventId();
    matmulInstr.sync.set_event_mode     = eventSetMode2Int(matmulWaveOp->gSetEventMode());

    m_WaveCode->writeInstruction(matmulInstr);
}

void
WaveCodeMatMul::generateLoadWeights(wave::MatMulWaveOp* matmulWaveOp)
{
    assert(matmulWaveOp->verify());
    if (matmulWaveOp->gWeightsOffsetInAtom() < 0) {
        return; // this MatMul reuses weights
    }
    //const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    //const wave::MatMulWaveOp::WaveId& waveId(matmulWaveOp->gWaveId());

    LDWEIGHTS ldweighsInstr;

    //TPB_CMD_HEADER  hdr;
    //setup_sync(ldweighsInstr.sync);
    const utils::DataType& dtype(matmulWaveOp->gInDtype());
    ldweighsInstr.dtype                 = dtype.gSimTypeId();
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
    const kcc_int64 addressInSbPart     = matmulWaveOp->gWeightsAtomId() * matmulWaveOp->gWaveAtomSize()
                                            + matmulWaveOp->gWeightsOffsetInAtom();

    ldweighsInstr.start_addr            = addressInSbPart + (matmulWaveOp->gOfmapCount() - 1) * dtype.gSizeInBytes();

    ldweighsInstr.x_step                = -1; // last column goes first, so decrement
    ldweighsInstr.x_num                 = matmulWaveOp->gOfmapCount();
    ldweighsInstr.num_row_partitions    = matmulWaveOp->gIfmapCount();

    m_WaveCode->writeInstruction(ldweighsInstr);
}


}}


