



#include "utils/inc/asserter.hpp"

#include "compisa/inc/compisatensortensorop.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/resaddwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecoderesadd.hpp"

namespace kcc {
namespace wavecode {

WaveCodeResAdd::WaveCodeResAdd(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeResAdd::generate(wave::WaveOp* waveOp)
{
    auto resaddWaveop = dynamic_cast<wave::ResAddWaveOp*>(waveOp);
    assert(resaddWaveop);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = resaddWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for Pool should be Pooling");

    Assert(resaddWaveop->qSrcAIsPsum() != resaddWaveop->qSrcBIsPsum(), "In ResAdd one source must be PSUM and another SB");

    compisa::TensorTensorOpInstr resaddInstr;

    TONGA_ISA_TPB_DTYPE& SrcADtype(resaddWaveop->qSrcAIsPsum() ? resaddInstr.in_psum_buf_dtype : resaddInstr.in_state_buf_dtype);
    TONGA_ISA_TPB_DTYPE& SrcBDtype(resaddWaveop->qSrcBIsPsum() ? resaddInstr.in_psum_buf_dtype : resaddInstr.in_state_buf_dtype);

    SrcADtype = resaddWaveop->gInADtype().gSimTypeId();
    SrcBDtype = resaddWaveop->gInBDtype().gSimTypeId();

    resaddInstr.out_dtype           = resaddWaveop->gOutDtype().gSimTypeId();

    resaddInstr.num_active_channels = resaddWaveop->gNumPartitions();
    if (resaddWaveop->gMultiply()) {    /* Hack in ResAdd to get Multiply to work with old ISA */
        resaddInstr.op = TONGA_ISA_TPB_ALU_OP_MULT;
    } else {
        resaddInstr.op = TONGA_ISA_TPB_ALU_OP_ADD;
    }


    // SrcA
    initMemAccess(resaddInstr.src_psum_buf_mem_pattern);
    initMemAccess(resaddInstr.src_state_buf_mem_pattern);


    TONGA_ISA_TPB_MEM_ACCESS_3D& SrcAPat(resaddWaveop->qSrcAIsPsum()
                                         ? resaddInstr.src_psum_buf_mem_pattern
                                         : resaddInstr.src_state_buf_mem_pattern);
    if (resaddWaveop->qSrcAIsPsum()) {
        resaddInstr.src_psum_buf_mem_pattern.start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcAPsumBankId(),
                                                                    resaddWaveop->gSrcAPsumBankOffset(),
                                                                    resaddWaveop->gInADtype());
    } else {
        resaddInstr.src_state_buf_mem_pattern.start_addr  = stateBuf.gEntryTpbAddress(0, /* row 0 */
                                                                    resaddWaveop->gSrcASbAddress());
    }
    SrcAPat.step_elem[0]    = resaddWaveop->gSrcAXStep();
    SrcAPat.num_elem[0]     = resaddWaveop->gSrcAXNum();
    SrcAPat.step_elem[1]    = resaddWaveop->gSrcAYStep();
    SrcAPat.num_elem[1]     = resaddWaveop->gSrcAYNum();
    SrcAPat.step_elem[2]    = resaddWaveop->gSrcAZStep();
    SrcAPat.num_elem[2]     = resaddWaveop->gSrcAZNum();

    // SrcB
    TONGA_ISA_TPB_MEM_ACCESS_3D& SrcBPat(resaddWaveop->qSrcBIsPsum()
                                         ? resaddInstr.src_psum_buf_mem_pattern
                                         : resaddInstr.src_state_buf_mem_pattern);
    if (resaddWaveop->qSrcBIsPsum()) {
        resaddInstr.src_psum_buf_mem_pattern.start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcBPsumBankId(),
                                                                    resaddWaveop->gSrcBPsumBankOffset(),
                                                                    resaddWaveop->gInBDtype());
    } else {
        resaddInstr.src_state_buf_mem_pattern.start_addr  = stateBuf.gEntryTpbAddress(0, /* row 0 */
                                                                    resaddWaveop->gSrcBSbAddress());
    }
    SrcBPat.step_elem[0]    = resaddWaveop->gSrcBXStep();
    SrcBPat.num_elem[0]     = resaddWaveop->gSrcBXNum();
    SrcBPat.step_elem[1]    = resaddWaveop->gSrcBYStep();
    SrcBPat.num_elem[1]     = resaddWaveop->gSrcBYNum();
    SrcBPat.step_elem[2]    = resaddWaveop->gSrcBZStep();
    SrcBPat.num_elem[2]     = resaddWaveop->gSrcBZNum();

    // Dst
    initMemAccess(resaddInstr.dst_mem_pattern);
    if (resaddWaveop->qDstIsPsum()) {
        resaddInstr.dst_mem_pattern.start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gDstPsumBankId(),
                                                                 resaddWaveop->gDstPsumBankOffset(),
                                                                 resaddWaveop->gOutDtype());
    } else {
        resaddInstr.dst_mem_pattern.start_addr  = stateBuf.gEntryTpbAddress(0, /* row 0 */
                                                            resaddWaveop->gDstSbAddress());
    }
    resaddInstr.dst_mem_pattern.step_elem[0]      = resaddWaveop->gDstXStep();
    resaddInstr.dst_mem_pattern.num_elem[0]       = resaddWaveop->gDstXNum();
    resaddInstr.dst_mem_pattern.step_elem[1]      = resaddWaveop->gDstYStep();
    resaddInstr.dst_mem_pattern.num_elem[1]       = resaddWaveop->gDstYNum();
    resaddInstr.dst_mem_pattern.step_elem[2]      = resaddWaveop->gDstZStep();
    resaddInstr.dst_mem_pattern.num_elem[2]       = resaddWaveop->gDstZNum();

    resaddInstr.inst_events.wait_event_idx     = 0;
    resaddInstr.inst_events.wait_event_mode    = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    resaddInstr.inst_events.set_event_idx      = 0;
    resaddInstr.inst_events.set_event_mode     = events::eventSetMode2Isa(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(resaddWaveop, resaddInstr.inst_events);
    } // end incoming events

    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(resaddWaveop, resaddInstr);
    }


    if (! instructionWritten) {
        m_WaveCode.writeInstruction(resaddInstr);
    }
}


}}


