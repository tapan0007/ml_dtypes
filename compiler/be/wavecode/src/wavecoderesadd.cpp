

#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "compisa/inc/compisatensortensorop.hpp"
#include "compisa/inc/compisatensorreduceop.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/resaddwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecoderesadd.hpp"

namespace kcc {
namespace wavecode {

/**********************************************************************
**********************************************************************/
WaveCodeResAdd::WaveCodeResAdd(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}


/**********************************************************************
**********************************************************************/
void
WaveCodeResAdd::generate(wave::WaveOp* waveOp)
{
    auto resaddWaveop = dynamic_cast<wave::ResAddWaveOp*>(waveOp);
    assert(resaddWaveop);

    if (resaddWaveop->qSrcAIsPsum() != resaddWaveop->qSrcBIsPsum()) {
        generateDiffBufSrc(resaddWaveop);
    } else {
        generateSameBufSrc(resaddWaveop);
    }
}

/**********************************************************************
**********************************************************************/
void
WaveCodeResAdd::generateDiffBufSrc(wave::ResAddWaveOp* resaddWaveop)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = resaddWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for ResAdd should be Pooling");
    Assert(resaddWaveop->qSrcAIsPsum() != resaddWaveop->qSrcBIsPsum(), "Sources for ResAdd must come from PSUM and SB, not from one");

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
    compisa::TensorTensorOpInstr tensortensorInstr;
    const utils::DataType& srcADtype(resaddWaveop->gInADtype());
    const utils::DataType& srcBDtype(resaddWaveop->gInBDtype());

    TONGA_ISA_TPB_MEM_ACCESS_3D& srcAPat(tensortensorInstr.src_mem_pattern[srcAIdx]);
    TONGA_ISA_TPB_MEM_ACCESS_3D& srcBPat(tensortensorInstr.src_mem_pattern[srcBIdx]);
    initMemAccess(srcAPat);
    initMemAccess(srcBPat);

    tensortensorInstr.in_dtype[srcAIdx] = srcADtype.gSimTypeId();
    tensortensorInstr.in_dtype[srcBIdx] = srcBDtype.gSimTypeId();

    if (resaddWaveop->qSrcAIsPsum()) {
        srcAPat.start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcAPsumBankId(),
                                                       resaddWaveop->gSrcAPsumBankOffset(),
                                                       srcADtype);
    } else {
        srcAPat.start_addr  = stateBuf.gEntryTpbAddress(
                                        arch.gNumberPeArrayRows()/2 * resaddWaveop->gSrcAStartAtMidPart(),
                                        resaddWaveop->gSrcASbAddress());
    }

    AssignWithSizeCheck(srcAPat.step_elem[PatDim_X], resaddWaveop->gSrcAXStep());
    AssignWithSizeCheck(srcAPat.num_elem[PatDim_X], resaddWaveop->gSrcAXNum());
    AssignWithSizeCheck(srcAPat.step_elem[PatDim_Y], resaddWaveop->gSrcAYStep());
    AssignWithSizeCheck(srcAPat.num_elem[PatDim_Y], resaddWaveop->gSrcAYNum());
    AssignWithSizeCheck(srcAPat.step_elem[PatDim_Z], resaddWaveop->gSrcAZStep());
    AssignWithSizeCheck(srcAPat.num_elem[PatDim_Z], resaddWaveop->gSrcAZNum());

    if (resaddWaveop->qSrcBIsPsum()) {
        srcBPat.start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcBPsumBankId(),
                                                       resaddWaveop->gSrcBPsumBankOffset(),
                                                       srcBDtype);
    } else {
        srcBPat.start_addr  = stateBuf.gEntryTpbAddress(
                                        arch.gNumberPeArrayRows()/2 * resaddWaveop->gSrcBStartAtMidPart(),
                                        resaddWaveop->gSrcBSbAddress());
    }

    AssignWithSizeCheck(srcBPat.step_elem[PatDim_X], resaddWaveop->gSrcBXStep());
    AssignWithSizeCheck(srcBPat.num_elem[PatDim_X], resaddWaveop->gSrcBXNum());
    AssignWithSizeCheck(srcBPat.step_elem[PatDim_Y], resaddWaveop->gSrcBYStep());
    AssignWithSizeCheck(srcBPat.num_elem[PatDim_Y], resaddWaveop->gSrcBYNum());
    AssignWithSizeCheck(srcBPat.step_elem[PatDim_Z], resaddWaveop->gSrcBZStep());
    AssignWithSizeCheck(srcBPat.num_elem[PatDim_Z], resaddWaveop->gSrcBZNum());

    //**********************************************************************

    tensortensorInstr.num_active_channels = resaddWaveop->gNumPartitions();
    if (resaddWaveop->gMultiply()) {
        tensortensorInstr.op = TONGA_ISA_TPB_ALU_OP_MULT;
    } else {
        tensortensorInstr.op = TONGA_ISA_TPB_ALU_OP_ADD;
    }


    //-----------------------------------------------------------------
    // Dst
    tensortensorInstr.out_dtype           = resaddWaveop->gOutDtype().gSimTypeId();
    TONGA_ISA_TPB_MEM_ACCESS_3D& DstPat(tensortensorInstr.dst_mem_pattern);
    initMemAccess(DstPat);
    if (resaddWaveop->qDstIsPsum()) {
        DstPat.start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gDstPsumBankId(),
                                                      resaddWaveop->gDstPsumBankOffset(),
                                                      resaddWaveop->gOutDtype());
    } else {
        DstPat.start_addr  = stateBuf.gEntryTpbAddress(
                                        arch.gNumberPeArrayRows()/2 * resaddWaveop->gDstStartAtMidPart(),
                                        resaddWaveop->gDstSbAddress());
    }
    AssignWithSizeCheck(DstPat.step_elem[PatDim_X], resaddWaveop->gDstXStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_X], resaddWaveop->gDstXNum());
    AssignWithSizeCheck(DstPat.step_elem[PatDim_Y], resaddWaveop->gDstYStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_Y], resaddWaveop->gDstYNum());
    AssignWithSizeCheck(DstPat.step_elem[PatDim_Z], resaddWaveop->gDstZStep());
    AssignWithSizeCheck(DstPat.num_elem[PatDim_Z], resaddWaveop->gDstZNum());

    //-----------------------------------------------------------------
    tensortensorInstr.inst_events.wait_event_idx     = 0;
    tensortensorInstr.inst_events.wait_event_mode    = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    tensortensorInstr.inst_events.set_event_idx      = 0;
    tensortensorInstr.inst_events.set_event_mode     = events::eventSetMode2Isa(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(resaddWaveop, tensortensorInstr.inst_events);
    } // end incoming events

    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(resaddWaveop, tensortensorInstr);
    }
    if (! instructionWritten) {
        std::ostringstream oss;
        oss << resaddWaveop->gOrder() << "-" << resaddWaveop->gName();
        m_WaveCode.SaveName(tensortensorInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(tensortensorInstr);
    }
}

/**********************************************************************
**********************************************************************/
void
WaveCodeResAdd::generateSameBufSrc(wave::ResAddWaveOp* resaddWaveop)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = resaddWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for ResAdd should be Pooling");

    //-----------------------------------------------------------------
    compisa::TensorReduceOpInstr tensorReduceInstr;

    Assert(resaddWaveop->gInADtype().gSimTypeId() == resaddWaveop->gInBDtype().gSimTypeId(),
        "ResAdd waveop that has sources in one buffer cannot have different data types");
    Assert(resaddWaveop->gSrcAXNum() == resaddWaveop->gSrcBXNum(),
        "When tensors for ResAdd are in the same buffer, X count must be equal");
    Assert(resaddWaveop->gSrcAYNum() == resaddWaveop->gSrcBYNum(),
        "When tensors for ResAdd are in the same buffer, Y count must be equal");
    Assert(resaddWaveop->gSrcAZNum() == resaddWaveop->gSrcBZNum(),
        "When tensors for ResAdd are in the same buffer, Z count must be equal");
    Assert(resaddWaveop->gSrcAXStep() == resaddWaveop->gSrcBXStep(),
        "When tensors for ResAdd are in the same buffer, X step must be equal");
    Assert(resaddWaveop->gSrcAYStep() == resaddWaveop->gSrcBYStep(),
        "When tensors for ResAdd are in the same buffer, Y step must be equal");
    Assert(resaddWaveop->gSrcAZStep() == resaddWaveop->gSrcBZStep(),
        "When tensors for ResAdd are in the same buffer, Z step must be equal");

    //-----------------------------------------------------------------
    const utils::DataType& inDtype(resaddWaveop->gInADtype());
    tensorReduceInstr.in_dtype  = inDtype.gSimTypeId();
    tensorReduceInstr.out_dtype = resaddWaveop->gOutDtype().gSimTypeId();
    tensorReduceInstr.num_active_channels = resaddWaveop->gNumPartitions();
    if (resaddWaveop->gMultiply()) {
        tensorReduceInstr.op = TONGA_ISA_TPB_ALU_OP_MULT;
    } else {
        tensorReduceInstr.op = TONGA_ISA_TPB_ALU_OP_ADD;
    }

    //-----------------------------------------------------------------
    tensorReduceInstr.op_dim = TONGA_ISA_TPB_TENSOR_SUBDIM_X;

    //-----------------------------------------------------------------
    TONGA_ISA_TPB_MEM_ACCESS_4D& srcPat(tensorReduceInstr.src_mem_pattern);
    initMemAccess(srcPat);

    kcc_int64 addrA;
    kcc_int64 addrB;
    if (resaddWaveop->qSrcAIsPsum()) {
        addrA = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcAPsumBankId(),
                                                         resaddWaveop->gSrcAPsumBankOffset(),
                                                         resaddWaveop->gInADtype());
        addrB = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcBPsumBankId(),
                                                         resaddWaveop->gSrcBPsumBankOffset(),
                                                         resaddWaveop->gInBDtype());
    } else {
        addrA = stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * resaddWaveop->gSrcAStartAtMidPart(),
                                          resaddWaveop->gSrcASbAddress());
        addrB = stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * resaddWaveop->gSrcBStartAtMidPart(),
                                          resaddWaveop->gSrcBSbAddress());
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
            "ResAdd from same buffer: delta address is not multiple of data size");

    AssignWithSizeCheck(srcPat.step_elem[PatDim_X], deltaAddr/inDtypeSize);
    AssignWithSizeCheck(srcPat.num_elem[PatDim_X], 2); // reduction of 2 tensors is on X
    AssignWithSizeCheck(srcPat.step_elem[PatDim_Y], resaddWaveop->gSrcAXStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_Y], resaddWaveop->gSrcAXNum());
    AssignWithSizeCheck(srcPat.step_elem[PatDim_Z], resaddWaveop->gSrcAYStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_Z], resaddWaveop->gSrcAYNum());
    AssignWithSizeCheck(srcPat.step_elem[PatDim_W], resaddWaveop->gSrcAZStep());
    AssignWithSizeCheck(srcPat.num_elem[PatDim_W], resaddWaveop->gSrcAZNum());

    //-----------------------------------------------------------------
    TONGA_ISA_TPB_MEM_ACCESS_4D& dstPat(tensorReduceInstr.dst_mem_pattern);
    initMemAccess(dstPat);

    if (resaddWaveop->qDstIsPsum()) {
        dstPat.start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gDstPsumBankId(),
                                                      resaddWaveop->gDstPsumBankOffset(),
                                                      resaddWaveop->gOutDtype());
    } else {
        dstPat.start_addr  = stateBuf.gEntryTpbAddress(
                                        arch.gNumberPeArrayRows()/2 * resaddWaveop->gDstStartAtMidPart(),
                                        resaddWaveop->gDstSbAddress());
    }

    AssignWithSizeCheck(dstPat.step_elem[PatDim_X], 0); // Destination pattern should be 3D
    AssignWithSizeCheck(dstPat.num_elem[PatDim_X], 1);
    AssignWithSizeCheck(dstPat.step_elem[PatDim_Y], resaddWaveop->gDstXStep());
    AssignWithSizeCheck(dstPat.num_elem[PatDim_Y], resaddWaveop->gDstXNum());
    AssignWithSizeCheck(dstPat.step_elem[PatDim_Z], resaddWaveop->gDstYStep());
    AssignWithSizeCheck(dstPat.num_elem[PatDim_Z], resaddWaveop->gDstYNum());
    AssignWithSizeCheck(dstPat.step_elem[PatDim_W], resaddWaveop->gDstZStep());
    AssignWithSizeCheck(dstPat.num_elem[PatDim_W], resaddWaveop->gDstZNum());


    //-----------------------------------------------------------------
    tensorReduceInstr.inst_events.wait_event_idx     = 0;
    tensorReduceInstr.inst_events.wait_event_mode    = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    tensorReduceInstr.inst_events.set_event_idx      = 0;
    tensorReduceInstr.inst_events.set_event_mode     = events::eventSetMode2Isa(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(resaddWaveop, tensorReduceInstr.inst_events);
    } // end incoming events

    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(resaddWaveop, tensorReduceInstr);
    }
    if (! instructionWritten) {
        std::ostringstream oss;
        oss << resaddWaveop->gOrder() << "-" << resaddWaveop->gName();
        m_WaveCode.SaveName(tensorReduceInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(tensorReduceInstr);
    }
}

}}


