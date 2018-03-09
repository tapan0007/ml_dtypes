#include "shared/inc/tpb_isa_activate.hpp"

#include "utils/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/activationwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodeactivation.hpp"

namespace kcc {
namespace wavecode {

WaveCodeActivation::WaveCodeActivation(WaveCode* waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeActivation::generate(wave::WaveOp* waveOp)
{
    auto activationWaveOp = dynamic_cast<wave::ActivationWaveOp*>(waveOp);
    assert(activationWaveOp);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    ACTIVATION activationInstr;

    activationInstr.activation_func     = activationWaveOp->gSimActivationFunc();
    activationInstr.in_dtype            = activationWaveOp->gInDtype().gSimTypeId();
    activationInstr.bias_dtype          = activationWaveOp->gBiasDtype().gSimTypeId();
    activationInstr.out_dtype           = activationWaveOp->gOutDtype().gSimTypeId();

    // TODO: for now Activation reads from 0 elem in bank.
    activationInstr.src_start_addr      = psumBuf.gEntryTpbAddress(activationWaveOp->gSrcPsumBankId(), 0, activationWaveOp->gInDtype());

    activationInstr.src_x_step          = activationWaveOp->gSrcXStep();
    activationInstr.src_y_step          = activationWaveOp->gSrcYStep();
    // activationInstr.src_z_step          = activationWaveOp->gSrcZStep(); // when available in the new ISA
    activationInstr.src_x_num           = activationWaveOp->gSrcXNum();
    activationInstr.src_y_num           = activationWaveOp->gSrcYNum();
    // activationInstr.src_z_num           = activationWaveOp->gSrcZNum(); // when available in the new ISA

    if (activationWaveOp->qDstIsPsum()) {
        activationInstr.dst_start_addr  = psumBuf.gEntryTpbAddress(activationWaveOp->gDstPsumBankId(),
                                                                  0, /* bank offset 0 */
                                                                  activationWaveOp->gOutDtype());
    } else {
        activationInstr.dst_start_addr  = stateBuf.gEntryTpbAddress(0, /* row 0 */
                                                activationWaveOp->gDstSbAtomId() * activationWaveOp->gWaveAtomSize()
                                                    + activationWaveOp->gDstSbOffsetInAtom());
    }
    activationInstr.dst_x_step      = activationWaveOp->gDstXStep();
    activationInstr.dst_y_step      = activationWaveOp->gDstYStep();
    activationInstr.dst_z_step      = activationWaveOp->gDstZStep();
    activationInstr.dst_x_num       = activationWaveOp->gDstXNum();
    activationInstr.dst_y_num       = activationWaveOp->gDstYNum();
    activationInstr.dst_z_num       = activationWaveOp->gDstZNum();

    activationInstr.scale_value         = activationWaveOp->gScale();
    if (activationWaveOp->qBiasAddEn ()) {
        activationInstr.acc_addr        = stateBuf.gEntryTpbAddress(
                                            0,   //row 0 for now
                                            activationWaveOp->gBiasAtomId() * activationWaveOp->gWaveAtomSize()
                                                + activationWaveOp->gBiasOffsetInAtom());
    } else {
        activationInstr.acc_addr        = stateBuf.gAllZeroOffsetTpbAddress();
    }
    activationInstr.num_partitions      = activationWaveOp->gNumPartitions();

    activationInstr.sync.set_event_id       = activationWaveOp->gSetEventId();
    activationInstr.sync.set_event_mode     = eventSetMode2Int(activationWaveOp->gSetEventMode());
    activationInstr.sync.wait_event_id      = activationWaveOp->gWaitEventId();
    activationInstr.sync.wait_event_mode    = eventWaitMode2Int(activationWaveOp->gWaitEventMode());

    m_WaveCode->writeInstruction(activationInstr);
}


}}


