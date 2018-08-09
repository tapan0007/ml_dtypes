#include "utils/inc/debug.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/misc.hpp"

#include "compisa/inc/compisaset.hpp"
#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisaclear.hpp"
#include "compisa/inc/compisanop.hpp"
#include "compisa/inc/compisamatmul.hpp"
#include "compisa/inc/compisasimrdnpy.hpp"


#include "compisa/inc/compisaldweights.hpp"

#include "compisa/inc/compisapool.hpp"

#include "compisa/inc/compisaactivate.hpp"


#include "compisa/inc/compisawrite.hpp"
#include "compisa/inc/compisatensortensorop.hpp"
#include "compisa/inc/compisatensorscalarop.hpp"
#include "compisa/inc/compisatensorscalarptrop.hpp"
#include "compisa/inc/compisatensorreduceop.hpp"
#include "compisa/inc/compisacopy.hpp"
#include "compisa/inc/compisacast.hpp"
#include "compisa/inc/compisamemset.hpp"
#include "compisa/inc/compisaregload.hpp"
#include "compisa/inc/compisaregshuffle.hpp"
#include "compisa/inc/compisaregstore.hpp"

#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisasimwrnpy.hpp"
#include "compisa/inc/compisasimrdnpy.hpp"

#include "compisa/inc/compisadmatrigger.hpp"
#include "compisa/inc/compisasimdmacopy.hpp"


#include "wave/inc/waveedge.hpp"
#include "wavecode/inc/wavecode.hpp"

namespace kcc {
namespace wavecode {


/***********************************************************************
 * PE Array
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::MatMulInstr>(const compisa::MatMulInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PeArrayInstrStream);
    m_PeArrayPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::LdWeightsInstr>(const compisa::LdWeightsInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PeArrayInstrStream);
    m_PeArrayPc += instSize;
}

/***********************************************************************
 * Pooling Eng
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::PoolInstr>(const compisa::PoolInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}


template<>
void WaveCode::writeInstruction<compisa::TensorTensorOpInstr>(const compisa::TensorTensorOpInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::TensorScalarOpInstr>(const compisa::TensorScalarOpInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::TensorScalarPtrOpInstr>(const compisa::TensorScalarPtrOpInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::TensorReduceOpInstr>(const compisa::TensorReduceOpInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::CopyInstr>(const compisa::CopyInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::MemSetInstr>(const compisa::MemSetInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::CastInstr>(const compisa::CastInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::RegLoadInstr>(const compisa::RegLoadInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::RegStoreInstr>(const compisa::RegStoreInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::RegShuffleInstr>(const compisa::RegShuffleInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
    m_PoolEngPc += instSize;
}


/***********************************************************************
 * Activation Eng
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::ActivateInstr >(const compisa::ActivateInstr & instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_ActEngInstrStream);
    m_ActEngPc += instSize;
}





/***********************************************************************
 * DMA/Angel Eng
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::SimRdNpyInstr>(const compisa::SimRdNpyInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_DmaInstrStream);
    m_DmaPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::SimWrNpyInstr>(const compisa::SimWrNpyInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_DmaInstrStream);
    m_DmaPc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::SimMemCpyInstr>(const compisa::SimMemCpyInstr& instruction)
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    fwrite(&instruction, instSize, 1, m_InstrStreams->m_DmaInstrStream);
    m_DmaPc += instSize;
}







/***********************************************************************
 * Event related - Multiple Engines
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::WaitInstr>(const compisa::WaitInstr& instruction, EngineId engId)
{
    if (instruction.event_idx == 0x12 && qBinFileRuntimeKelf()) {
        utils::breakFunc(__LINE__);
    }
    Assert(qParallelStreams(), "Cannot generate wait-for-event instruction in serial mode");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    switch (engId) {
    case EngineId::Pooling:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
        m_PoolEngPc += instSize;
        break;
    case EngineId::PeArray:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PeArrayInstrStream);
        m_PeArrayPc += instSize;
        break;
    case EngineId::Activation:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_ActEngInstrStream);
        m_ActEngPc += instSize;
        break;
    case EngineId::StreamProc:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_StreamProcInstrStream);
        m_StreamProcPc += instSize;
        break;
    case EngineId::DmaEng:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_DmaInstrStream);
        m_DmaPc += instSize;
        break;
    default:
        Assert(false, "Wrong EngineId for Wait instruction: ", static_cast<int>(engId));
    }
}


template<>
void WaveCode::writeInstruction<compisa::SetInstr>(const compisa::SetInstr& setInstr, EngineId engId)
{
#if false
    const auto& instruction(setInstr);
#else
    compisa::NopInstr instruction;
    // TONGA_ISA_TPB_INST_EVENTS
    instruction.inst_events.wait_event_mode    = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    instruction.inst_events.wait_event_idx     = 0;
    instruction.inst_events.set_event_mode     = events::eventSetMode2Isa(events::EventSetMode::OnEndInstr);
    instruction.inst_events.set_event_idx      = setInstr.event_idx;
    instruction.cycle_cnt                      = 1;
    strcpy(
        reinterpret_cast<char*>(instruction.reserved),
        reinterpret_cast<const char*>(setInstr.reserved));
    instruction.reserved[ArraySizeof(instruction.reserved)-1] = '\0';
    // kcc::ArraySizeof(T (&)[N])
#endif

    Assert(qParallelStreams(), "Cannot generate set-event instruction in serial mode");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    switch (engId) {
    case EngineId::Pooling:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
        m_PoolEngPc += instSize;
        break;
    case EngineId::PeArray:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PeArrayInstrStream);
        m_PeArrayPc += instSize;
        break;
    case EngineId::Activation:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_ActEngInstrStream);
        m_ActEngPc += instSize;
        break;
    case EngineId::StreamProc:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_StreamProcInstrStream);
        m_StreamProcPc += instSize;
        break;
    case EngineId::DmaEng:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_DmaInstrStream);
        m_DmaPc += instSize;
        break;
    default:
        Assert(false, "Wrong EngineId for Set instruction: ", static_cast<int>(engId));
    }
}


template<>
void WaveCode::writeInstruction<compisa::ClearInstr>(const compisa::ClearInstr& instruction, EngineId engId)
{
    Assert(qParallelStreams(), "Cannot generate clear-event instruction in serial mode");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    switch (engId) {
    case EngineId::Pooling:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
        m_PoolEngPc += instSize;
        break;
    case EngineId::PeArray:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PeArrayInstrStream);
        m_PeArrayPc += instSize;
        break;
    case EngineId::Activation:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_ActEngInstrStream);
        m_ActEngPc += instSize;
        break;
    case EngineId::StreamProc:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_StreamProcInstrStream);
        m_StreamProcPc += instSize;
        break;
    case EngineId::DmaEng:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_DmaInstrStream);
        m_DmaPc += instSize;
        break;
    default:
        Assert(false, "Wrong EngineId for Clear instruction: ", static_cast<int>(engId));
    }
}


/***********************************************************************
 * Multiple Eng
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::WriteInstr>(const compisa::WriteInstr& instruction, EngineId engId)
{
    checkForNoSync(instruction.inst_events);

    switch (engId) {
    case EngineId::Pooling:
        fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PoolEngInstrStream);
        break;
    case EngineId::PeArray:
        fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PeArrayInstrStream);
        break;
    case EngineId::Activation:
        fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_ActEngInstrStream);
        break;
    case EngineId::StreamProc:
        fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_StreamProcInstrStream);
        break;
    case EngineId::DmaEng:
        fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_DmaInstrStream);
        break;
    default:
        Assert(false, "Wrong EngineId for Write instruction: ", static_cast<int>(engId));
    }
}


template<>
void WaveCode::writeInstruction<compisa::NopInstr>(const compisa::NopInstr& instruction, EngineId engId)
{
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    switch (engId) {
    case EngineId::Pooling:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
        m_PoolEngPc += instSize;
        break;
    case EngineId::PeArray:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PeArrayInstrStream);
        m_PeArrayPc += instSize;
        break;
    case EngineId::Activation:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_ActEngInstrStream);
        m_ActEngPc += instSize;
        break;
    case EngineId::StreamProc:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_StreamProcInstrStream);
        m_StreamProcPc += instSize;
        break;
    case EngineId::DmaEng:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_DmaInstrStream);
        m_DmaPc += instSize;
        break;
    default:
        Assert(false, "Wrong EngineId for Nop instruction: ", static_cast<int>(engId));
    }
}

template<>
void WaveCode::writeInstruction<compisa::DmaTriggerInstr>(const compisa::DmaTriggerInstr& instruction, EngineId engId)
{
    Assert(qBinFileRuntimeKelf(), "DmaTriggerInstr is available in RuntimeKelf binary only");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    switch (engId) {
    case EngineId::Pooling:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
        m_PoolEngPc += instSize;
        break;
    case EngineId::PeArray:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PeArrayInstrStream);
        m_PeArrayPc += instSize;
        break;
    case EngineId::Activation:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_ActEngInstrStream);
        m_ActEngPc += instSize;
        break;
    /*
    case EngineId::StreamProc:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_StreamProcInstrStream);
        m_StreamProcPc += instSize;
        break;
    */
    default:
        Assert(false, "Wrong EngineId for DmaTrigger instruction: ", static_cast<int>(engId));
    }
}


template<>
void WaveCode::writeInstruction<compisa::SimDmaCopyInstr>(const compisa::SimDmaCopyInstr& instruction, EngineId engId)
{
    Assert(qBinFileSimKelf(), "SimDmaCopy is available in SimKelf binary only");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    switch (engId) {
    case EngineId::Pooling:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PoolEngInstrStream);
        m_PoolEngPc += instSize;
        break;
    case EngineId::PeArray:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_PeArrayInstrStream);
        m_PeArrayPc += instSize;
        break;
    case EngineId::Activation:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_ActEngInstrStream);
        m_ActEngPc += instSize;
        break;
    /*
    case EngineId::StreamProc:
        fwrite(&instruction, instSize, 1, m_InstrStreams->m_StreamProcInstrStream);
        m_StreamProcPc += instSize;
        break;
    */
    default:
        Assert(false, "Wrong EngineId for DmaTrigger instruction: ", static_cast<int>(engId));
    }
}

}}


