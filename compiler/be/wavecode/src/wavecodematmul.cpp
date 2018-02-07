#include "tpb_isa_ldweights.hpp"
#include "tpb_isa_matmul.hpp"

#include "wave/inc/matmulwaveop.hpp"
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

    generateLoadWeights(matmulWaveOp, matmulWaveOp.qStart());
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
WaveCodeMatMul::generateLoadWeights(MatMullWaveOp *matmulWaveOp, bool firstWeight)
{
    LDWEIGHTS ldweighsInstr;

    TPB_CMD_SYNC         sync;
    if (firstWeight
    setup_sync(weight_args.sync, in_event, SET_EVENT_ON_END_RD_SRC);
                    setup_sync(weight_args.sync, matmul_args.sync.set_event_id, SET_EVENT_ON_END_RD_SRC);

    TPB_CMD_DEQUANT       dquant;
    uint32_t        start_addr = {0};
    int16_t         x_step = {0};      // 1's complement, granularity of dtype
    uint8_t         x_num  = {0};
    uint8_t         num_row_partitions = {0};
    LDWEIGHTS() : hdr(LDWEIGHTS_OPC, *this) {}
}

}}


