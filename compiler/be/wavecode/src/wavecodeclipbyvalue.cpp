
#include "compisa/inc/compisatensorscalarop.hpp"


#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/clipbyvaluewaveop.hpp"
//#include "wave/inc/matmulwaveop.hpp"
//#include "wave/inc/poolwaveop.hpp"
//#include "wave/inc/sbatomloadwaveop.hpp"
//#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodeclipbyvalue.hpp"

namespace kcc {
namespace wavecode {


WaveCodeClipByValue::WaveCodeClipByValue(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeClipByValue::generate(wave::WaveOp* waveop)
{
    auto clipByValueWaveop = dynamic_cast<wave::ClipByValueWaveOp*>(waveop);
    assert(clipByValueWaveop);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());
    const EngineId engineId = clipByValueWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for ClipByValue waveop should be Pooling engine");

    compisa::TensorScalarOpInstr tensorScalarOpInstr;

    tensorScalarOpInstr.in_dtype            = clipByValueWaveop->gInDtype().gSimTypeId();
    tensorScalarOpInstr.out_dtype           = clipByValueWaveop->gOutDtype().gSimTypeId();


    // TODO: for now Activation reads from 0 elem in bank.
    initMemAccess(tensorScalarOpInstr.src_mem_pattern);
    if (clipByValueWaveop->qSrcIsPsum()) {
        tensorScalarOpInstr.src_mem_pattern.start_addr  = psumBuf.gEntryTpbAddress(
                                                            clipByValueWaveop->gSrcPsumBankId(),
                                                            clipByValueWaveop->gSrcPsumBankOffset(),
                                                            clipByValueWaveop->gInDtype());
    } else {
        tensorScalarOpInstr.src_mem_pattern.start_addr  = stateBuf.gEntryTpbAddress(
                                                            arch.gNumberPeArrayRows()/2 * clipByValueWaveop->gSrcStartAtMidPart(),
                                                            clipByValueWaveop->gSrcSbAddress());
    }
    tensorScalarOpInstr.src_mem_pattern.step_elem[PatDim_X]    = clipByValueWaveop->gSrcXStep();
    tensorScalarOpInstr.src_mem_pattern.num_elem[PatDim_X]     = clipByValueWaveop->gSrcXNum();
    tensorScalarOpInstr.src_mem_pattern.step_elem[PatDim_Y]    = clipByValueWaveop->gSrcYStep();
    tensorScalarOpInstr.src_mem_pattern.num_elem[PatDim_Y]     = clipByValueWaveop->gSrcYNum();
    tensorScalarOpInstr.src_mem_pattern.step_elem[PatDim_Z]    = clipByValueWaveop->gSrcZStep();
    tensorScalarOpInstr.src_mem_pattern.num_elem[PatDim_Z]     = clipByValueWaveop->gSrcZNum();


    initMemAccess(tensorScalarOpInstr.dst_mem_pattern);
    if (clipByValueWaveop->qDstIsPsum()) {
        tensorScalarOpInstr.dst_mem_pattern.start_addr  = psumBuf.gEntryTpbAddress(
                                                                  clipByValueWaveop->gDstPsumBankId(),
                                                                  clipByValueWaveop->gDstPsumBankOffset(),
                                                                  clipByValueWaveop->gOutDtype());
    } else {
        tensorScalarOpInstr.dst_mem_pattern.start_addr  = stateBuf.gEntryTpbAddress(
                                                            arch.gNumberPeArrayRows()/2 * clipByValueWaveop->gDstStartAtMidPart(),
                                                            clipByValueWaveop->gDstSbAddress());
    }
    tensorScalarOpInstr.dst_mem_pattern.step_elem[PatDim_X]    = clipByValueWaveop->gDstXStep();
    tensorScalarOpInstr.dst_mem_pattern.num_elem[PatDim_X]     = clipByValueWaveop->gDstXNum();
    tensorScalarOpInstr.dst_mem_pattern.step_elem[PatDim_Y]    = clipByValueWaveop->gDstYStep();
    tensorScalarOpInstr.dst_mem_pattern.num_elem[PatDim_Y]     = clipByValueWaveop->gDstYNum();
    tensorScalarOpInstr.dst_mem_pattern.step_elem[PatDim_Z]    = clipByValueWaveop->gDstZStep();
    tensorScalarOpInstr.dst_mem_pattern.num_elem[PatDim_Z]     = clipByValueWaveop->gDstZNum();

    tensorScalarOpInstr.num_active_channels = clipByValueWaveop->gNumPartitions();

    tensorScalarOpInstr.op[0] = TONGA_ISA_TPB_ALU_OP_MIN;
    tensorScalarOpInstr.imm_val[0] = clipByValueWaveop->gMaxValue();

    tensorScalarOpInstr.op[1] = TONGA_ISA_TPB_ALU_OP_MAX;
    tensorScalarOpInstr.imm_val[1] = clipByValueWaveop->gMinValue();



    tensorScalarOpInstr.inst_events.wait_event_idx  = 0;
    tensorScalarOpInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    tensorScalarOpInstr.inst_events.set_event_idx   = 0;
    tensorScalarOpInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(clipByValueWaveop, tensorScalarOpInstr.inst_events);
    } // incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(clipByValueWaveop, tensorScalarOpInstr);
    }


    if (! instructionWritten) {
        std::ostringstream oss;
        oss << clipByValueWaveop->gOrder() << "-" <<  clipByValueWaveop->gName();
        m_WaveCode.SaveName(tensorScalarOpInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(tensorScalarOpInstr);
    }
}


}}

