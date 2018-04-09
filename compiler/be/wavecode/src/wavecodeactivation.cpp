
#include "compisa/inc/compisaactivation.hpp"


#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodeactivation.hpp"

namespace kcc {
namespace wavecode {


WaveCodeActivation::WaveCodeActivation(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeActivation::generate(wave::WaveOp* waveop)
{
    auto activationWaveop = dynamic_cast<wave::ActivationWaveOp*>(waveop);
    assert(activationWaveop);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());
    const EngineId engineId = activationWaveop->gEngineId();
    Assert(EngineId::Activation == engineId, "Engine id for Activation waveop should be Activation");

    compisa::ActivationInstr activationInstr;

    activationInstr.activation_func     = activationWaveop->gSimActivationFunc();
    activationInstr.in_dtype            = activationWaveop->gInDtype().gSimTypeId();
    activationInstr.bias_dtype          = activationWaveop->gBiasDtype().gSimTypeId();
    activationInstr.out_dtype           = activationWaveop->gOutDtype().gSimTypeId();

    // TODO: for now Activation reads from 0 elem in bank.
    activationInstr.src_start_addr      = psumBuf.gEntryTpbAddress(activationWaveop->gSrcPsumBankId(), 0, activationWaveop->gInDtype());

    activationInstr.src_x_step          = activationWaveop->gSrcXStep();
    activationInstr.src_y_step          = activationWaveop->gSrcYStep();
    // activationInstr.src_z_step          = activationWaveop->gSrcZStep(); // when available in the new ISA
    activationInstr.src_x_num           = activationWaveop->gSrcXNum();
    activationInstr.src_y_num           = activationWaveop->gSrcYNum();
    // activationInstr.src_z_num           = activationWaveop->gSrcZNum(); // when available in the new ISA

    if (activationWaveop->qDstIsPsum()) {
        activationInstr.dst_start_addr  = psumBuf.gEntryTpbAddress(activationWaveop->gDstPsumBankId(),
                                                                  0, /* bank offset 0 */
                                                                  activationWaveop->gOutDtype());
    } else {
        activationInstr.dst_start_addr  = stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * activationWaveop->gDstStartAtMidPart(),
                                                activationWaveop->gDstSbAddress());
    }
    activationInstr.dst_x_step      = activationWaveop->gDstXStep();
    activationInstr.dst_y_step      = activationWaveop->gDstYStep();
    activationInstr.dst_z_step      = activationWaveop->gDstZStep();
    activationInstr.dst_x_num       = activationWaveop->gDstXNum();
    activationInstr.dst_y_num       = activationWaveop->gDstYNum();
    activationInstr.dst_z_num       = activationWaveop->gDstZNum();

    activationInstr.scale_value         = activationWaveop->gScale();
    if (activationWaveop->qBiasAddEn ()) {
        activationInstr.acc_addr        = stateBuf.gEntryTpbAddress(
                                            arch.gNumberPeArrayRows()/2 * activationWaveop->gBiasStartAtMidPart(),
                                            activationWaveop->gBiasSbAddress());
    } else {
        activationInstr.acc_addr        = stateBuf.gAllZeroOffsetTpbAddress(activationWaveop->gBiasDtype());
    }
    activationInstr.num_partitions      = activationWaveop->gNumPartitions();

    activationInstr.sync.wait_event_id    = 0;
    activationInstr.sync.wait_event_mode  = events::eventWaitMode2Int(events::EventWaitMode::DontWait);
    activationInstr.sync.set_event_id    = 0;
    activationInstr.sync.set_event_mode  = events::eventSetMode2Int(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(activationWaveop, activationInstr.sync);
    } // incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(activationWaveop, activationInstr);
    }



    if (! instructionWritten) {
        m_WaveCode.writeInstruction(activationInstr);
    }
}


}}


