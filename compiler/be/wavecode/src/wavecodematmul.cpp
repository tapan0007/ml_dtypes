#include "tpb_isa_ldweights.hpp"
#include "tpb_isa_matmul.hpp"

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

    generateLoadWeights(matmulWaveOp, matmulWaveOp->qStart());
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
WaveCodeMatMul::generateLoadWeights(wave::MatMulWaveOp* matmulWaveOp, bool /*firstWeight*/)
{
    assert(matmulWaveOp->verify());
    if (matmulWaveOp->gWeightsOffsetInAtom() < 0) {
        return; // this MatMul reuses weights
    }
    const wave::MatMulWaveOp::WaveId& waveId(matmulWaveOp->gWaveId());

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
    ldweighsInstr.start_addr            = -1; // TODO
    ldweighsInstr.x_step                = -1; // last column goes first, so decr
    ldweighsInstr.x_num                 = waveId.gTileNumColumns();
    ldweighsInstr.num_row_partitions    = -1; // TODO

    m_WaveCode->writeInstruction(ldweighsInstr);
}


}}


