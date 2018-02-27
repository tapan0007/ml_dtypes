#include "tpb_isa_ldweights.hpp"
#include "tpb_isa_activate.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/resaddwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecoderesadd.hpp"

namespace kcc {
namespace wavecode {

WaveCodeResAdd::WaveCodeResAdd(WaveCode* waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeResAdd::generate(wave::WaveOp* waveOp)
{
    auto resaddWaveOp = dynamic_cast<wave::ResAddWaveOp*>(waveOp);
    assert(resaddWaveOp);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    MATADD resaddInstr;

    resaddInstr.in_a_dtype          = resaddWaveOp->gInADtype().gSimTypeId();
    resaddInstr.in_b_dtype          = resaddWaveOp->gInBDtype().gSimTypeId();
    resaddInstr.out_dtype           = resaddWaveOp->gOutDtype().gSimTypeId();
    resaddInstr.num_partitions      = resaddWaveOp->gNumPartitions();

    // SrcA
    if (resaddWaveOp->qSrcAIsPsum()) {
        resaddInstr.src_a_start_addr  = psumBuf.gEntryTpbAddress(resaddWaveOp->gSrcAPsumBankId(),
                                                                 resaddWaveOp->gSrcAPsumBankOffset(),
                                                                 resaddWaveOp->gInADtype());
    } else {
        resaddInstr.src_a_start_addr  = stateBuf.gEntryTpbAddress(0, /* row 0 */
                                                resaddWaveOp->gSrcASbAtomId() * resaddWaveOp->gWaveAtomSize()
                                                    + resaddWaveOp->gSrcASbOffsetInAtom());
    }
    resaddInstr.src_a_x_step      = resaddWaveOp->gSrcAXStep();
    resaddInstr.src_a_y_step      = resaddWaveOp->gSrcAYStep();
    resaddInstr.src_a_z_step      = resaddWaveOp->gSrcAZStep();
    resaddInstr.src_a_x_num       = resaddWaveOp->gSrcAXNum();
    resaddInstr.src_a_y_num       = resaddWaveOp->gSrcAYNum();
    resaddInstr.src_a_z_num       = resaddWaveOp->gSrcAZNum();

    // SrcB
    if (resaddWaveOp->qSrcBIsPsum()) {
        resaddInstr.src_b_start_addr  = psumBuf.gEntryTpbAddress(resaddWaveOp->gSrcBPsumBankId(),
                                                                 resaddWaveOp->gSrcBPsumBankOffset(),
                                                                 resaddWaveOp->gInADtype());
    } else {
        resaddInstr.src_b_start_addr  = stateBuf.gEntryTpbAddress(0, /* row 0 */
                                                resaddWaveOp->gSrcBSbAtomId() * resaddWaveOp->gWaveAtomSize()
                                                    + resaddWaveOp->gSrcBSbOffsetInAtom());
    }
    resaddInstr.src_b_x_step      = resaddWaveOp->gSrcBXStep();
    resaddInstr.src_b_y_step      = resaddWaveOp->gSrcBYStep();
    resaddInstr.src_b_z_step      = resaddWaveOp->gSrcBZStep();
    resaddInstr.src_b_x_num       = resaddWaveOp->gSrcBXNum();
    resaddInstr.src_b_y_num       = resaddWaveOp->gSrcBYNum();
    resaddInstr.src_b_z_num       = resaddWaveOp->gSrcBZNum();

    // Dst
    if (resaddWaveOp->qDstIsPsum()) {
        resaddInstr.dst_start_addr  = psumBuf.gEntryTpbAddress(resaddWaveOp->gDstPsumBankId(),
                                                                 resaddWaveOp->gDstPsumBankOffset(),
                                                                 resaddWaveOp->gInADtype());
    } else {
        resaddInstr.dst_start_addr  = stateBuf.gEntryTpbAddress(0, /* row 0 */
                                                resaddWaveOp->gDstSbAtomId() * resaddWaveOp->gWaveAtomSize()
                                                    + resaddWaveOp->gDstSbOffsetInAtom());
    }
    resaddInstr.dst_x_step      = resaddWaveOp->gDstXStep();
    resaddInstr.dst_y_step      = resaddWaveOp->gDstYStep();
    resaddInstr.dst_z_step      = resaddWaveOp->gDstZStep();
    resaddInstr.dst_x_num       = resaddWaveOp->gDstXNum();
    resaddInstr.dst_y_num       = resaddWaveOp->gDstYNum();
    resaddInstr.dst_z_num       = resaddWaveOp->gDstZNum();

    m_WaveCode->writeInstruction(resaddInstr);
}


}}


