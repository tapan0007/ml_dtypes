

#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "compisa/inc/compisatensorscalarptrop.hpp"
//#include "compisa/inc/compisatensorreduceop.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/tensorscalarptrwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodetensorscalarptr.hpp"

namespace kcc {
namespace wavecode {

/**********************************************************************
**********************************************************************/
WaveCodeTensorScalarPtr::WaveCodeTensorScalarPtr(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}


/**********************************************************************
**********************************************************************/
void
WaveCodeTensorScalarPtr::generate(wave::WaveOp* waveOp)
{
    auto tensorscalarWaveop = dynamic_cast<wave::TensorScalarPtrWaveOp*>(waveOp);
    assert(tensorscalarWaveop);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = tensorscalarWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for TensorScalarPtr should be Pooling");

    //-----------------------------------------------------------------
    // ARITH now
    compisa::TensorScalarPtrOpInstr tensorscalarInstr(TONGA_ISA_TPB_OPCODE_TENSOR_SCALAR_ARITH_OP);
    AssignWithSizeCheck(tensorscalarInstr.op[0], static_cast<TONGA_ISA_TPB_ALU_OP>(tensorscalarWaveop->gAluOp(0)));
    AssignWithSizeCheck(tensorscalarInstr.op[1], static_cast<TONGA_ISA_TPB_ALU_OP>(tensorscalarWaveop->gAluOp(1)));
    AssignWithSizeCheck(tensorscalarInstr.imm_ptr[0], tensorscalarWaveop->gImmPtr(0));
    AssignWithSizeCheck(tensorscalarInstr.imm_ptr[1], tensorscalarWaveop->gImmPtr(1));

    const utils::DataType& srcDtype(tensorscalarWaveop->gInDtype());

    TONGA_ISA_TPB_MEM_ACCESS_3D& srcPat(tensorscalarInstr.src_mem_pattern);
    initMemAccess(srcPat);

    AssignWithSizeCheck(tensorscalarInstr.in_dtype, srcDtype.gSimTypeId());

    if (tensorscalarWaveop->qSrcIsPsum()) {
        AssignWithSizeCheck(srcPat.start_addr,
                            psumBuf.gEntryTpbAddress(tensorscalarWaveop->gSrcPsumBankId(),
                                                     tensorscalarWaveop->gSrcPsumBankOffset(),
                                                     srcDtype));
    } else {
        AssignWithSizeCheck(srcPat.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * tensorscalarWaveop->gSrcStartAtMidPart(),
                                                      tensorscalarWaveop->gSrcSbAddress()));
    }

    AssignWithSizeCheck(srcPat.step_elem[PatDim_X], tensorscalarWaveop->gSrcXStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_X], tensorscalarWaveop->gSrcXNum());
    AssignWithSizeCheck(srcPat.step_elem[PatDim_Y], tensorscalarWaveop->gSrcYStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_Y], tensorscalarWaveop->gSrcYNum());
    AssignWithSizeCheck(srcPat.step_elem[PatDim_Z], tensorscalarWaveop->gSrcZStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_Z], tensorscalarWaveop->gSrcZNum());

    //**********************************************************************

    AssignWithSizeCheck(tensorscalarInstr.num_active_channels, tensorscalarWaveop->gNumPartitions());




    //-----------------------------------------------------------------
    // Dst
    AssignWithSizeCheck(tensorscalarInstr.out_dtype, tensorscalarWaveop->gOutDtype().gSimTypeId());
    TONGA_ISA_TPB_MEM_ACCESS_3D& DstPat(tensorscalarInstr.dst_mem_pattern);
    initMemAccess(DstPat);
    if (tensorscalarWaveop->qDstIsPsum()) {
        AssignWithSizeCheck(DstPat.start_addr,
                            psumBuf.gEntryTpbAddress(tensorscalarWaveop->gDstPsumBankId(),
                                                 tensorscalarWaveop->gDstPsumBankOffset(),
                                                 tensorscalarWaveop->gOutDtype()));
    } else {
        AssignWithSizeCheck(DstPat.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * tensorscalarWaveop->gDstStartAtMidPart(),
                                                  tensorscalarWaveop->gDstSbAddress()));
    }
    AssignWithSizeCheck(DstPat.step_elem[PatDim_X], tensorscalarWaveop->gDstXStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_X], tensorscalarWaveop->gDstXNum());
    AssignWithSizeCheck(DstPat.step_elem[PatDim_Y], tensorscalarWaveop->gDstYStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_Y], tensorscalarWaveop->gDstYNum());
    AssignWithSizeCheck(DstPat.step_elem[PatDim_Z], tensorscalarWaveop->gDstZStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_Z], tensorscalarWaveop->gDstZNum());

    //-----------------------------------------------------------------
    AssignWithSizeCheck(tensorscalarInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(tensorscalarInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(tensorscalarInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(tensorscalarInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(tensorscalarWaveop, tensorscalarInstr.inst_events);
    } // end incoming events

    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(tensorscalarWaveop, tensorscalarInstr);
    }
    if (! instructionWritten) {
        std::ostringstream oss;
        oss << tensorscalarWaveop->gOrder() << "-" << tensorscalarWaveop->gName();
        m_WaveCode.SaveName(tensorscalarInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(tensorscalarInstr);
    }
}

}}


