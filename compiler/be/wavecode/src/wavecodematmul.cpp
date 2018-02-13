#include "tpb_isa_ldweights.hpp"
#include "tpb_isa_matmul.hpp"

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

/*
    {
      "ifmaps_atom_id": 48,
      "ifmaps_offset_in_atom": 0,
      "layer_name": "1conv/i1",
      "previous_waveops": [
        "1conv/i1/SBAtomFile_0",
        "input/SBAtomFile_0"
      ],
      "psum_bank_id": 0,
      "start": true,
      "wave_id": [ 0, 0, 0, 0, 0, 0, 0 ],
      "wave_id_format": "nmhwcrs",
      "waveop_name": "1conv/i1/MatMul_n0_m0_h0_w0_c0_r0_s0",
      "waveop_type": "MatMul",
      "weights_atom_id": 0,
      "weights_offset_in_atom": 0
    },
*/


void
WaveCodeMatMul::generateMatMul(wave::MatMulWaveOp* matmulWaveOp)
{
    const layers::ConvLayer* const convLayer = matmulWaveOp->gConvLayer();
    const arch::Arch& arch(convLayer->gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());

    MATMUL matmulInstr;
    matmulInstr.dtype                   = matmulWaveOp->gDataType().gTypeId();
    matmulInstr.num_row_partitions      = matmulWaveOp->gIfmapCount();
    matmulInstr.num_column_partitions   = matmulWaveOp->gOfmapCount();

    matmulInstr.fmap_start_addr         = matmulWaveOp->gIfmapsAtomId() * wave::MatMulWaveOp::AtomSize +
                                          matmulWaveOp->gIfmapsOffsetInAtom();
    matmulInstr.fmap_x_num              = matmulWaveOp->gIfmapTileWidth();
    matmulInstr.fmap_x_step             = convLayer->gStrideLeftRight();
    matmulInstr.fmap_y_num              = matmulWaveOp->gIfmapTileHeight();
    matmulInstr.fmap_y_step             = matmulWaveOp->gIfmapTileWidth() * convLayer->gStrideTopBottom();
    matmulInstr.fmap_z_num              = 1; /* no batching right now */
    matmulInstr.fmap_z_step             = 0;

    matmulInstr.psum_start_addr         = psumBuf.gEntryTpbAddress(matmulWaveOp->gPsumBankId(), matmulWaveOp->gPsumBankOffset());
    matmulInstr.psum_x_num              = matmulWaveOp->gOfmapTileWidth();
  //matmul_args.psum_x_num = matmul_args.fmap_x_num;
    matmulInstr.psum_x_step             = 1;
    matmulInstr.psum_y_num              = matmulWaveOp->gOfmapTileHeight();
  //matmul_args.psum_y_num = matmul_args.fmap_y_num;
    matmulInstr.psum_y_step             = matmulInstr.psum_x_num;

    matmulInstr.start_tensor_calc       = matmulWaveOp->qStartTensorCalc();
    matmulInstr.stop_tensor_calc        = matmulWaveOp->qStopTensorCalc();
    //setup_sync(matmulInstr.sync, -1, SET_EVENT_ON_END_WR_DST);

    m_WaveCode->writeInstruction(matmulInstr, WaveCode::UseStream_PeArray);
}

void
WaveCodeMatMul::generateLoadWeights(wave::MatMulWaveOp* matmulWaveOp)
{
    assert(matmulWaveOp->verify());
    const layers::ConvLayer* const convLayer = matmulWaveOp->gConvLayer();
    if (matmulWaveOp->gWeightsOffsetInAtom() < 0) {
        return; // this MatMul reuses weights
    }
    //const wave::MatMulWaveOp::WaveId& waveId(matmulWaveOp->gWaveId());

    LDWEIGHTS ldweighsInstr;

    //TPB_CMD_HEADER  hdr;
    //setup_sync(ldweighsInstr.sync);
    ldweighsInstr.dtype                 = matmulWaveOp->gDataType().gTypeId();
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
    const kcc_int32 sizeofWeights = matmulWaveOp->gWeightsOffsetInAtom() +
                                          (matmulWaveOp->gWeightsAtomId() *
                                           convLayer->gWaveAtomSize());
    ldweighsInstr.start_addr            = m_WaveCode->gCurrentDramAddress(sizeofWeights);
    ldweighsInstr.x_step                = -1; // last column goes first, so decrement
    ldweighsInstr.x_num                 = matmulWaveOp->gNumOfmapsInFold();
    ldweighsInstr.num_row_partitions    = matmulWaveOp->gIfmapCount();

    m_WaveCode->writeInstruction(ldweighsInstr, WaveCode::UseStream_PeArray);
}


}}


