

#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "compisa/inc/compisatensorscalarop.hpp"
#include "compisa/inc/compisatensorreduceop.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/tensorscalarconstwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodetensorscalarconst.hpp"

namespace kcc {
namespace wavecode {

/**********************************************************************
**********************************************************************/
WaveCodeTensorScalarConst::WaveCodeTensorScalarConst(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}


/**********************************************************************
**********************************************************************/
void
WaveCodeTensorScalarConst::generate(wave::WaveOp* waveOp)
{
    auto tensorscalarconstWaveop = dynamic_cast<wave::TensorScalarConstWaveOp*>(waveOp);
    assert(tensorscalarconstWaveop);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = tensorscalarconstWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for TensorScalarConst should be Pooling");

    //-----------------------------------------------------------------
    compisa::TensorScalarOpInstr tensorscalarInstr;
    const utils::DataType& srcDtype(tensorscalarconstWaveop->gInDtype());

    TONGA_ISA_TPB_MEM_ACCESS_3D& srcPat(tensorscalarInstr.src_mem_pattern);
    initMemAccess(srcPat);

    tensorscalarInstr.in_dtype = srcDtype.gSimTypeId();

    if (tensorscalarconstWaveop->qSrcIsPsum()) {
        srcPat.start_addr  = psumBuf.gEntryTpbAddress(tensorscalarconstWaveop->gSrcPsumBankId(),
                                                       tensorscalarconstWaveop->gSrcPsumBankOffset(),
                                                       srcDtype);
    } else {
        srcPat.start_addr  = stateBuf.gEntryTpbAddress(
                                        arch.gNumberPeArrayRows()/2 * tensorscalarconstWaveop->gSrcStartAtMidPart(),
                                        tensorscalarconstWaveop->gSrcSbAddress());
    }

    AssignWithSizeCheck(srcPat.step_elem[PatDim_X], tensorscalarconstWaveop->gSrcXStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_X], tensorscalarconstWaveop->gSrcXNum());
    AssignWithSizeCheck(srcPat.step_elem[PatDim_Y], tensorscalarconstWaveop->gSrcYStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_Y], tensorscalarconstWaveop->gSrcYNum());
    AssignWithSizeCheck(srcPat.step_elem[PatDim_Z], tensorscalarconstWaveop->gSrcZStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_Z], tensorscalarconstWaveop->gSrcZNum());

    //**********************************************************************

    tensorscalarInstr.num_active_channels = tensorscalarconstWaveop->gNumPartitions();

    tensorscalarInstr.op[0] = static_cast<TONGA_ISA_TPB_ALU_OP>(tensorscalarconstWaveop->gAluOp(0));
    tensorscalarInstr.op[1] = static_cast<TONGA_ISA_TPB_ALU_OP>(tensorscalarconstWaveop->gAluOp(1));
    tensorscalarInstr.imm_val[0] = static_cast<TONGA_ISA_TPB_ALU_OP>(tensorscalarconstWaveop->gImmVal(0));
    tensorscalarInstr.imm_val[1] = static_cast<TONGA_ISA_TPB_ALU_OP>(tensorscalarconstWaveop->gImmVal(1));



    //-----------------------------------------------------------------
    // Dst
    tensorscalarInstr.out_dtype           = tensorscalarconstWaveop->gOutDtype().gSimTypeId();
    TONGA_ISA_TPB_MEM_ACCESS_3D& DstPat(tensorscalarInstr.dst_mem_pattern);
    initMemAccess(DstPat);
    if (tensorscalarconstWaveop->qDstIsPsum()) {
        DstPat.start_addr  = psumBuf.gEntryTpbAddress(tensorscalarconstWaveop->gDstPsumBankId(),
                                                      tensorscalarconstWaveop->gDstPsumBankOffset(),
                                                      tensorscalarconstWaveop->gOutDtype());
    } else {
        DstPat.start_addr  = stateBuf.gEntryTpbAddress(
                                        arch.gNumberPeArrayRows()/2 * tensorscalarconstWaveop->gDstStartAtMidPart(),
                                        tensorscalarconstWaveop->gDstSbAddress());
    }
    AssignWithSizeCheck(DstPat.step_elem[PatDim_X], tensorscalarconstWaveop->gDstXStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_X], tensorscalarconstWaveop->gDstXNum());
    AssignWithSizeCheck(DstPat.step_elem[PatDim_Y], tensorscalarconstWaveop->gDstYStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_Y], tensorscalarconstWaveop->gDstYNum());
    AssignWithSizeCheck(DstPat.step_elem[PatDim_Z], tensorscalarconstWaveop->gDstZStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_Z], tensorscalarconstWaveop->gDstZNum());

    //-----------------------------------------------------------------
    tensorscalarInstr.inst_events.wait_event_idx     = 0;
    tensorscalarInstr.inst_events.wait_event_mode    = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    tensorscalarInstr.inst_events.set_event_idx      = 0;
    tensorscalarInstr.inst_events.set_event_mode     = events::eventSetMode2Isa(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(tensorscalarconstWaveop, tensorscalarInstr.inst_events);
    } // end incoming events

    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(tensorscalarconstWaveop, tensorscalarInstr);
    }
    if (! instructionWritten) {
        std::ostringstream oss;
        oss << tensorscalarconstWaveop->gOrder() << "-" << tensorscalarconstWaveop->gName();
        m_WaveCode.SaveName(tensorscalarInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(tensorscalarInstr);
    }
}

}}


