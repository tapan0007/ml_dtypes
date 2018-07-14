

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

    //-----------------------------------------------------------------
    compisa::TensorTensorOpInstr tensortensorInstr;

    Assert(resaddWaveop->qSrcAIsPsum() != resaddWaveop->qSrcBIsPsum(), "Sources for ResAdd must come from PSUM and SB, not from one");
    TONGA_ISA_TPB_DTYPE& SrcADtype(resaddWaveop->qSrcAIsPsum() ? tensortensorInstr.in_dtype[0] : tensortensorInstr.in_dtype[1]);
    TONGA_ISA_TPB_DTYPE& SrcBDtype(resaddWaveop->qSrcBIsPsum() ? tensortensorInstr.in_dtype[0] : tensortensorInstr.in_dtype[1]);

    //-----------------------------------------------------------------
    SrcADtype = resaddWaveop->gInADtype().gSimTypeId();
    SrcBDtype = resaddWaveop->gInBDtype().gSimTypeId();

    tensortensorInstr.out_dtype           = resaddWaveop->gOutDtype().gSimTypeId();

    tensortensorInstr.num_active_channels = resaddWaveop->gNumPartitions();
    if (resaddWaveop->gMultiply()) {
        tensortensorInstr.op = TONGA_ISA_TPB_ALU_OP_MULT;
    } else {
        tensortensorInstr.op = TONGA_ISA_TPB_ALU_OP_ADD;
    }


    //-----------------------------------------------------------------
    TONGA_ISA_TPB_MEM_ACCESS_3D& SrcAPat(resaddWaveop->qSrcAIsPsum()
                                         ? tensortensorInstr.src_mem_pattern[0]
                                         : tensortensorInstr.src_mem_pattern[1]);
    initMemAccess(SrcAPat);
    TONGA_ISA_TPB_MEM_ACCESS_3D& SrcBPat(resaddWaveop->qSrcBIsPsum()
                                         ? tensortensorInstr.src_mem_pattern[0]
                                         : tensortensorInstr.src_mem_pattern[1]);
    initMemAccess(SrcBPat);

    //-----------------------------------------------------------------
    // SrcA
    if (resaddWaveop->qSrcAIsPsum()) {
        SrcAPat.start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcAPsumBankId(),
                                                       resaddWaveop->gSrcAPsumBankOffset(),
                                                       resaddWaveop->gInADtype());
    } else {
        SrcAPat.start_addr  = stateBuf.gEntryTpbAddress(
                                        arch.gNumberPeArrayRows()/2 * resaddWaveop->gSrcAStartAtMidPart(),
                                        resaddWaveop->gSrcASbAddress());
    }
    SrcAPat.step_elem[PatDim_X]    = resaddWaveop->gSrcAXStep();
    SrcAPat.num_elem[PatDim_X]     = resaddWaveop->gSrcAXNum();
    SrcAPat.step_elem[PatDim_Y]    = resaddWaveop->gSrcAYStep();
    SrcAPat.num_elem[PatDim_Y]     = resaddWaveop->gSrcAYNum();
    SrcAPat.step_elem[PatDim_Z]    = resaddWaveop->gSrcAZStep();
    SrcAPat.num_elem[PatDim_Z]     = resaddWaveop->gSrcAZNum();

    //-----------------------------------------------------------------
    // SrcB
    if (resaddWaveop->qSrcBIsPsum()) {
        SrcBPat.start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcBPsumBankId(),
                                                       resaddWaveop->gSrcBPsumBankOffset(),
                                                       resaddWaveop->gInBDtype());
    } else {
        SrcBPat.start_addr  = stateBuf.gEntryTpbAddress(
                                        arch.gNumberPeArrayRows()/2 * resaddWaveop->gSrcBStartAtMidPart(),
                                        resaddWaveop->gSrcBSbAddress());
    }
    SrcBPat.step_elem[PatDim_X]    = resaddWaveop->gSrcBXStep();
    SrcBPat.num_elem[PatDim_X]     = resaddWaveop->gSrcBXNum();
    SrcBPat.step_elem[PatDim_Y]    = resaddWaveop->gSrcBYStep();
    SrcBPat.num_elem[PatDim_Y]     = resaddWaveop->gSrcBYNum();
    SrcBPat.step_elem[PatDim_Z]    = resaddWaveop->gSrcBZStep();
    SrcBPat.num_elem[PatDim_Z]     = resaddWaveop->gSrcBZNum();

    //-----------------------------------------------------------------
    // Dst
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
    DstPat.step_elem[PatDim_X]      = resaddWaveop->gDstXStep();
    DstPat.num_elem[PatDim_X]       = resaddWaveop->gDstXNum();
    DstPat.step_elem[PatDim_Y]      = resaddWaveop->gDstYStep();
    DstPat.num_elem[PatDim_Y]       = resaddWaveop->gDstYNum();
    DstPat.step_elem[PatDim_Z]      = resaddWaveop->gDstZStep();
    DstPat.num_elem[PatDim_Z]       = resaddWaveop->gDstZNum();

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

    srcPat.step_elem[PatDim_X]  = deltaAddr/inDtypeSize;
    srcPat.num_elem[PatDim_X]   = 2; // reduction of 2 tensors is on X
    srcPat.step_elem[PatDim_Y]  = resaddWaveop->gSrcAXStep();
    srcPat.num_elem[PatDim_Y]   = resaddWaveop->gSrcAXNum();
    srcPat.step_elem[PatDim_Z]  = resaddWaveop->gSrcAYStep();
    srcPat.num_elem[PatDim_Z]   = resaddWaveop->gSrcAYNum();
    srcPat.step_elem[PatDim_W]  = resaddWaveop->gSrcAZStep();
    srcPat.num_elem[PatDim_W]   = resaddWaveop->gSrcAZNum();

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

    dstPat.step_elem[PatDim_X]  = 0; // Destination pattern should be 3D
    dstPat.num_elem[PatDim_X]   = 1;
    dstPat.step_elem[PatDim_Y]  = resaddWaveop->gDstXStep();
    dstPat.num_elem[PatDim_Y]   = resaddWaveop->gDstXNum();
    dstPat.step_elem[PatDim_Z]  = resaddWaveop->gDstYStep();
    dstPat.num_elem[PatDim_Z]   = resaddWaveop->gDstYNum();
    dstPat.step_elem[PatDim_W]  = resaddWaveop->gDstZStep();
    dstPat.num_elem[PatDim_W]   = resaddWaveop->gDstZNum();


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


