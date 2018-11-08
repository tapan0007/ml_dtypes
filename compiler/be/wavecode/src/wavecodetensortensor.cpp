

#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "compisa/inc/compisatensortensorop.hpp"
#include "compisa/inc/compisatensorreduceop.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/tensortensorwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodetensortensor.hpp"

namespace kcc {
namespace wavecode {

/**********************************************************************
**********************************************************************/
WaveCodeTensorTensor::WaveCodeTensorTensor(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}


/**********************************************************************
**********************************************************************/
void
WaveCodeTensorTensor::generate(wave::WaveOp* waveOp)
{
    auto tensortensorWaveop = dynamic_cast<wave::TensorTensorWaveOp*>(waveOp);
    assert(tensortensorWaveop);

    if (tensortensorWaveop->qSrcAIsPsum() != tensortensorWaveop->qSrcBIsPsum()) {
        generateDiffBufSrc(tensortensorWaveop);
    } else {
        generateSameBufSrc(tensortensorWaveop);
    }
}

/**********************************************************************
**********************************************************************/
void
WaveCodeTensorTensor::generateDiffBufSrc(wave::TensorTensorWaveOp* tensortensorWaveop)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = tensortensorWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for TensorTensor should be Pooling");
    Assert(tensortensorWaveop->qSrcAIsPsum() != tensortensorWaveop->qSrcBIsPsum(), "Sources for TensorTensor must come from PSUM and SB, not from one");

    //-----------------------------------------------------------------
    // In ISA: for TensorTensor instruction and op OP, the calculated value is
    //     (src[0] OP src[1]).
    // !!! It is NOT
    //     (src[1] OP src[0]).
    //
    // In wavegraph.json the calculated value is
    //     A OP B
    // It is not
    //     B - A
    enum {
        srcAIdx = 0,
        srcBIdx = 1,
    };
    //-----------------------------------------------------------------
    // For now ARITH tensor-tensor
    compisa::TensorTensorOpInstr tensortensorInstr(TONGA_ISA_TPB_OPCODE_TENSOR_TENSOR_ARITH_OP);
    const utils::DataType& srcADtype(tensortensorWaveop->gInADtype());
    const utils::DataType& srcBDtype(tensortensorWaveop->gInBDtype());
    TONGA_ISA_TPB_MEM_ACCESS_3D& srcAPat(tensortensorInstr.src_mem_pattern[srcAIdx]);
    TONGA_ISA_TPB_MEM_ACCESS_3D& srcBPat(tensortensorInstr.src_mem_pattern[srcBIdx]);

    initMemAccess(srcAPat);
    initMemAccess(srcBPat);

    AssignWithSizeCheck(tensortensorInstr.in_dtype[srcAIdx], srcADtype.gSimTypeId());
    AssignWithSizeCheck(tensortensorInstr.in_dtype[srcBIdx], srcBDtype.gSimTypeId());

    if (tensortensorWaveop->qSrcAIsPsum()) {
        AssignWithSizeCheck(srcAPat.start_addr,
                            psumBuf.gEntryTpbAddress(tensortensorWaveop->gSrcAPsumBankId(),
                                                     tensortensorWaveop->gSrcAPsumBankOffset(),
                                                     srcADtype));
    } else {
        AssignWithSizeCheck(srcAPat.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * tensortensorWaveop->gSrcAStartAtMidPart(),
                                                      tensortensorWaveop->gSrcASbAddress()));
    }

    AssignWithSizeCheck(srcAPat.step_elem[PatDim_X], tensortensorWaveop->gSrcAXStep());
    AssignWithSizeCheck(srcAPat.num_elem[PatDim_X], tensortensorWaveop->gSrcAXNum());
    AssignWithSizeCheck(srcAPat.step_elem[PatDim_Y], tensortensorWaveop->gSrcAYStep());
    AssignWithSizeCheck(srcAPat.num_elem[PatDim_Y], tensortensorWaveop->gSrcAYNum());
    AssignWithSizeCheck(srcAPat.step_elem[PatDim_Z], tensortensorWaveop->gSrcAZStep());
    AssignWithSizeCheck(srcAPat.num_elem[PatDim_Z], tensortensorWaveop->gSrcAZNum());

    if (tensortensorWaveop->qSrcBIsPsum()) {
        AssignWithSizeCheck(srcBPat.start_addr,
                            psumBuf.gEntryTpbAddress(tensortensorWaveop->gSrcBPsumBankId(),
                                                     tensortensorWaveop->gSrcBPsumBankOffset(),
                                                     srcBDtype));
    } else {
        AssignWithSizeCheck(srcBPat.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * tensortensorWaveop->gSrcBStartAtMidPart(),
                                                      tensortensorWaveop->gSrcBSbAddress()));
    }

    AssignWithSizeCheck(srcBPat.step_elem[PatDim_X], tensortensorWaveop->gSrcBXStep());
    AssignWithSizeCheck(srcBPat.num_elem[PatDim_X], tensortensorWaveop->gSrcBXNum());
    AssignWithSizeCheck(srcBPat.step_elem[PatDim_Y], tensortensorWaveop->gSrcBYStep());
    AssignWithSizeCheck(srcBPat.num_elem[PatDim_Y], tensortensorWaveop->gSrcBYNum());
    AssignWithSizeCheck(srcBPat.step_elem[PatDim_Z], tensortensorWaveop->gSrcBZStep());
    AssignWithSizeCheck(srcBPat.num_elem[PatDim_Z], tensortensorWaveop->gSrcBZNum());

    //**********************************************************************

    AssignWithSizeCheck(tensortensorInstr.num_active_channels, tensortensorWaveop->gNumPartitions());
    AssignWithSizeCheck(tensortensorInstr.op, static_cast<TONGA_ISA_TPB_ALU_OP>(tensortensorWaveop->gAluOp()));

    //-----------------------------------------------------------------
    // Dst
    AssignWithSizeCheck(tensortensorInstr.out_dtype, tensortensorWaveop->gOutDtype().gSimTypeId());
    TONGA_ISA_TPB_MEM_ACCESS_3D& DstPat(tensortensorInstr.dst_mem_pattern);
    initMemAccess(DstPat);
    if (tensortensorWaveop->qDstIsPsum()) {
        AssignWithSizeCheck(DstPat.start_addr,
                            psumBuf.gEntryTpbAddress(tensortensorWaveop->gDstPsumBankId(),
                                                     tensortensorWaveop->gDstPsumBankOffset(),
                                                     tensortensorWaveop->gOutDtype()));
    } else {
        AssignWithSizeCheck(DstPat.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * tensortensorWaveop->gDstStartAtMidPart(),
                                                      tensortensorWaveop->gDstSbAddress()));
    }
    AssignWithSizeCheck(DstPat.step_elem[PatDim_X], tensortensorWaveop->gDstXStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_X], tensortensorWaveop->gDstXNum());
    AssignWithSizeCheck(DstPat.step_elem[PatDim_Y], tensortensorWaveop->gDstYStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_Y], tensortensorWaveop->gDstYNum());
    AssignWithSizeCheck(DstPat.step_elem[PatDim_Z], tensortensorWaveop->gDstZStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_Z], tensortensorWaveop->gDstZNum());

    //-----------------------------------------------------------------
    AssignWithSizeCheck(tensortensorInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(tensortensorInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(tensortensorInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(tensortensorInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(tensortensorWaveop, tensortensorInstr.inst_events);
    } // end incoming events

    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(tensortensorWaveop, tensortensorInstr);
    }
    if (! instructionWritten) {
        std::ostringstream oss;
        oss << tensortensorWaveop->gOrder() << "-" << tensortensorWaveop->gName();
        m_WaveCode.SaveName(tensortensorInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(tensortensorInstr);
    }
}

/**********************************************************************
**********************************************************************/
void
WaveCodeTensorTensor::generateSameBufSrc(wave::TensorTensorWaveOp* tensortensorWaveop)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = tensortensorWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for TensorTensor should be Pooling");

    //-----------------------------------------------------------------
    //For now ARITH
    compisa::TensorReduceOpInstr tensorReduceInstr(TONGA_ISA_TPB_OPCODE_TENSOR_REDUCE_ARITH_OP);

    Assert(tensortensorWaveop->gInADtype().gSimTypeId() == tensortensorWaveop->gInBDtype().gSimTypeId(),
        "TensorTensor waveop that has sources in one buffer cannot have different data types");
    Assert(tensortensorWaveop->gSrcAXNum() == tensortensorWaveop->gSrcBXNum(),
        "When tensors for TensorTensor are in the same buffer, X count must be equal");
    Assert(tensortensorWaveop->gSrcAYNum() == tensortensorWaveop->gSrcBYNum(),
        "When tensors for TensorTensor are in the same buffer, Y count must be equal");
    Assert(tensortensorWaveop->gSrcAZNum() == tensortensorWaveop->gSrcBZNum(),
        "When tensors for TensorTensor are in the same buffer, Z count must be equal");
    Assert(tensortensorWaveop->gSrcAXStep() == tensortensorWaveop->gSrcBXStep(),
        "When tensors for TensorTensor are in the same buffer, X step must be equal");
    Assert(tensortensorWaveop->gSrcAYStep() == tensortensorWaveop->gSrcBYStep(),
        "When tensors for TensorTensor are in the same buffer, Y step must be equal");
    Assert(tensortensorWaveop->gSrcAZStep() == tensortensorWaveop->gSrcBZStep(),
        "When tensors for TensorTensor are in the same buffer, Z step must be equal");

    //-----------------------------------------------------------------
    const utils::DataType& inDtype(tensortensorWaveop->gInADtype());
    AssignWithSizeCheck(tensorReduceInstr.in_dtype, inDtype.gSimTypeId());
    AssignWithSizeCheck(tensorReduceInstr.out_dtype, tensortensorWaveop->gOutDtype().gSimTypeId());
    AssignWithSizeCheck(tensorReduceInstr.num_active_channels, tensortensorWaveop->gNumPartitions());
    AssignWithSizeCheck(tensorReduceInstr.op, static_cast<TONGA_ISA_TPB_ALU_OP>(tensortensorWaveop->gAluOp()));

    //-----------------------------------------------------------------
    AssignWithSizeCheck(tensorReduceInstr.op_dim, TONGA_ISA_TPB_TENSOR_SUBDIM_X);

    //-----------------------------------------------------------------
    TONGA_ISA_TPB_MEM_ACCESS_4D& srcPat(tensorReduceInstr.src_mem_pattern);
    initMemAccess(srcPat);

    kcc_int64 addrA;
    kcc_int64 addrB;
    if (tensortensorWaveop->qSrcAIsPsum()) {
        AssignWithSizeCheck(addrA, psumBuf.gEntryTpbAddress(tensortensorWaveop->gSrcAPsumBankId(),
                                                         tensortensorWaveop->gSrcAPsumBankOffset(),
                                                         tensortensorWaveop->gInADtype()));
        AssignWithSizeCheck(addrB, psumBuf.gEntryTpbAddress(tensortensorWaveop->gSrcBPsumBankId(),
                                                         tensortensorWaveop->gSrcBPsumBankOffset(),
                                                         tensortensorWaveop->gInBDtype()));
    } else {
        AssignWithSizeCheck(addrA, stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * tensortensorWaveop->gSrcAStartAtMidPart(),
                                          tensortensorWaveop->gSrcASbAddress()));
        AssignWithSizeCheck(addrB, stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * tensortensorWaveop->gSrcBStartAtMidPart(),
                                          tensortensorWaveop->gSrcBSbAddress()));
    }
    kcc_int64 deltaAddr;
    if (addrA < addrB) {
        srcPat.start_addr  = addrA;
        deltaAddr = addrB - addrA;
    } else {
        srcPat.start_addr  = addrB;
        deltaAddr = addrA - addrB;
    }
    const kcc_int32 inDtypeSize = inDtype.gSizeInBytes();
    Assert((deltaAddr % inDtypeSize) == 0,
            "TensorTensor from same buffer: delta address is not multiple of data size");

    AssignWithSizeCheck(srcPat.step_elem[PatDim_X], deltaAddr/inDtypeSize);
    AssignWithSizeCheck(srcPat.num_elem[PatDim_X], 2); // reduction of 2 tensors is on X
    AssignWithSizeCheck(srcPat.step_elem[PatDim_Y], tensortensorWaveop->gSrcAXStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_Y], tensortensorWaveop->gSrcAXNum());
    AssignWithSizeCheck(srcPat.step_elem[PatDim_Z], tensortensorWaveop->gSrcAYStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_Z], tensortensorWaveop->gSrcAYNum());
    AssignWithSizeCheck(srcPat.step_elem[PatDim_W], tensortensorWaveop->gSrcAZStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_W], tensortensorWaveop->gSrcAZNum());

    //-----------------------------------------------------------------
    TONGA_ISA_TPB_MEM_ACCESS_4D& dstPat(tensorReduceInstr.dst_mem_pattern);
    initMemAccess(dstPat);

    if (tensortensorWaveop->qDstIsPsum()) {
        dstPat.start_addr  = psumBuf.gEntryTpbAddress(tensortensorWaveop->gDstPsumBankId(),
                                                      tensortensorWaveop->gDstPsumBankOffset(),
                                                      tensortensorWaveop->gOutDtype());
    } else {
        dstPat.start_addr  = stateBuf.gEntryTpbAddress(
                                        arch.gNumberPeArrayRows()/2 * tensortensorWaveop->gDstStartAtMidPart(),
                                        tensortensorWaveop->gDstSbAddress());
    }

    AssignWithSizeCheck(dstPat.step_elem[PatDim_X], 0); // Destination pattern should be 3D
    AssignWithSizeCheck(dstPat.num_elem[PatDim_X], 1);
    AssignWithSizeCheck(dstPat.step_elem[PatDim_Y], tensortensorWaveop->gDstXStep());
    AssignWithSizeCheck(dstPat.num_elem[PatDim_Y], tensortensorWaveop->gDstXNum());
    AssignWithSizeCheck(dstPat.step_elem[PatDim_Z], tensortensorWaveop->gDstYStep());
    AssignWithSizeCheck(dstPat.num_elem[PatDim_Z], tensortensorWaveop->gDstYNum());
    AssignWithSizeCheck(dstPat.step_elem[PatDim_W], tensortensorWaveop->gDstZStep());
    AssignWithSizeCheck(dstPat.num_elem[PatDim_W], tensortensorWaveop->gDstZNum());


    //-----------------------------------------------------------------
    AssignWithSizeCheck(tensorReduceInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(tensorReduceInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(tensorReduceInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(tensorReduceInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(tensortensorWaveop, tensorReduceInstr.inst_events);
    } // end incoming events

    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(tensortensorWaveop, tensorReduceInstr);
    }
    if (! instructionWritten) {
        std::ostringstream oss;
        oss << tensortensorWaveop->gOrder() << "-" << tensortensorWaveop->gName();
        m_WaveCode.SaveName(tensorReduceInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(tensorReduceInstr);
    }
}

}}


