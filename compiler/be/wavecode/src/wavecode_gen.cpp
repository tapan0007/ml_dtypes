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
#include "compisa/inc/compisasemaphore.hpp"

#include "compisa/inc/compisadmatrigger.hpp"

#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisasimwrnpy.hpp"
#include "compisa/inc/compisasimrdnpy.hpp"

#include "compisa/inc/compisasimdmacopy.hpp"


#include "wave/inc/waveedge.hpp"
#include "wavecode/inc/wavecode.hpp"

// WAIT in ActEng does not work. SIM https://issues.amazon.com/tonga-1489
#define NOP_FOR_WAIT_SET 1

namespace kcc {
namespace wavecode {


/***********************************************************************
 * PE Array
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::MatMulInstr>(const compisa::MatMulInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PeArray;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::LdWeightsInstr>(const compisa::LdWeightsInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PeArray;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

/***********************************************************************
 * Pooling Eng
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::PoolInstr>(const compisa::PoolInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}


template<>
void WaveCode::writeInstruction<compisa::TensorTensorOpInstr>(const compisa::TensorTensorOpInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::TensorScalarOpInstr>(const compisa::TensorScalarOpInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::TensorScalarPtrOpInstr>(const compisa::TensorScalarPtrOpInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::TensorReduceOpInstr>(const compisa::TensorReduceOpInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::CopyInstr>(const compisa::CopyInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::MemSetInstr>(const compisa::MemSetInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::CastInstr>(const compisa::CastInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::RegLoadInstr>(const compisa::RegLoadInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::RegStoreInstr>(const compisa::RegStoreInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::RegShuffleInstr>(const compisa::RegShuffleInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_PoolEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}


/***********************************************************************
 * Activation Eng
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::ActivateInstr >(const compisa::ActivateInstr & instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_ActEng;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}





/***********************************************************************
 * DMA/Angel Eng
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::SimRdNpyInstr>(const compisa::SimRdNpyInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_Angel;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::SimWrNpyInstr>(const compisa::SimWrNpyInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_Angel;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::SimMemCpyInstr>(const compisa::SimMemCpyInstr& instruction) const
{
    instruction.CheckValidity();
    checkForNoSync(instruction.inst_events);

    const kcc_int32 instSize = sizeof(instruction);
    auto engInfo = &m_InstrStreams->m_Angel;
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}







/***********************************************************************
 * Event related - Multiple Engines
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::WaitInstr>(const compisa::WaitInstr& waitInstr, EngineId engId) const
{
#if ! NOP_FOR_WAIT_SET
    const auto& instruction(waitInstr);
#else
    compisa::NopInstr instruction;
    instruction.inst_events.wait_event_mode    = events::eventWaitMode2Isa(events::EventWaitMode::WaitThenClear);
    instruction.inst_events.wait_event_idx     = waitInstr.event_idx;
    instruction.inst_events.set_event_mode     = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    instruction.inst_events.set_event_idx      = 0;

    instruction.cycle_cnt                      = 1;
    strcpy(
        reinterpret_cast<char*>(instruction.reserved),
        reinterpret_cast<const char*>(waitInstr.reserved));
    instruction.reserved[ArraySizeof(instruction.reserved)-1] = '\0';
#endif

    Assert(qParallelStreams(), "Cannot generate wait-for-event instruction in serial mode");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    InstrStreams::OneEngInfo* engInfo = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engInfo = &m_InstrStreams->m_PoolEng;
        break;
    case EngineId::PeArray:
        engInfo = &m_InstrStreams->m_PeArray;
        break;
    case EngineId::Activation:
        engInfo = &m_InstrStreams->m_ActEng;
        break;
    case EngineId::StreamProc:
        engInfo = &m_InstrStreams->m_StreamProc;
        break;
    case EngineId::AngelEng:
        engInfo = &m_InstrStreams->m_Angel;
        break;
    default:
        Assert(false, "Wrong EngineId for Wait instruction: ", static_cast<int>(engId));
    }
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}


template<>
void WaveCode::writeInstruction<compisa::SetInstr>(const compisa::SetInstr& setInstr, EngineId engId) const
{
#if ! NOP_FOR_WAIT_SET
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


    InstrStreams::OneEngInfo* engInfo = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engInfo = &m_InstrStreams->m_PoolEng;
        break;
    case EngineId::PeArray:
        engInfo = &m_InstrStreams->m_PeArray;
        break;
    case EngineId::Activation:
        engInfo = &m_InstrStreams->m_ActEng;
        break;
    case EngineId::StreamProc:
        engInfo = &m_InstrStreams->m_StreamProc;
        break;
    case EngineId::AngelEng:
        engInfo = &m_InstrStreams->m_Angel;
        break;
    default:
        Assert(false, "Wrong EngineId for Set instruction: ", static_cast<int>(engId));
    }
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}


template<>
void WaveCode::writeInstruction<compisa::ClearInstr>(const compisa::ClearInstr& instruction, EngineId engId) const
{
    Assert(qParallelStreams(), "Cannot generate clear-event instruction in serial mode");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    InstrStreams::OneEngInfo* engInfo = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engInfo = &m_InstrStreams->m_PoolEng;
        break;
    case EngineId::PeArray:
        engInfo = &m_InstrStreams->m_PeArray;
        break;
    case EngineId::Activation:
        engInfo = &m_InstrStreams->m_ActEng;
        break;
    case EngineId::StreamProc:
        engInfo = &m_InstrStreams->m_StreamProc;
        break;
    case EngineId::AngelEng:
        engInfo = &m_InstrStreams->m_Angel;
        break;
    default:
        Assert(false, "Wrong EngineId for Clear instruction: ", static_cast<int>(engId));
    }
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}


/***********************************************************************
 * Multiple Eng
***********************************************************************/
template<>
void WaveCode::writeInstruction<compisa::WriteInstr>(const compisa::WriteInstr& instruction, EngineId engId) const
{
    checkForNoSync(instruction.inst_events);
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    InstrStreams::OneEngInfo* engInfo = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engInfo = &m_InstrStreams->m_PoolEng;
        break;
    case EngineId::PeArray:
        engInfo = &m_InstrStreams->m_PeArray;
        break;
    case EngineId::Activation:
        engInfo = &m_InstrStreams->m_ActEng;
        break;
    case EngineId::StreamProc:
        engInfo = &m_InstrStreams->m_StreamProc;
        break;
    default:
        Assert(false, "Wrong EngineId for Write instruction: ", static_cast<int>(engId));
    }
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}


template<>
void WaveCode::writeInstruction<compisa::NopInstr>(const compisa::NopInstr& instruction, EngineId engId) const
{
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    InstrStreams::OneEngInfo* engInfo = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engInfo = &m_InstrStreams->m_PoolEng;
        break;
    case EngineId::PeArray:
        engInfo = &m_InstrStreams->m_PeArray;
        break;
    case EngineId::Activation:
        engInfo = &m_InstrStreams->m_ActEng;
        break;
    case EngineId::StreamProc:
        engInfo = &m_InstrStreams->m_StreamProc;
        break;
    default:
        Assert(false, "Wrong EngineId for Nop instruction: ", static_cast<int>(engId));
    }
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::SemaphoreInstr >(const compisa::SemaphoreInstr & instruction, EngineId engId) const
{
    Assert(qBinFileRuntimeKelf(), "SemaphoreInstr is available in RuntimeKelf binary only");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    InstrStreams::OneEngInfo* engInfo = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engInfo = &m_InstrStreams->m_PoolEng;
        break;
    case EngineId::PeArray:
        engInfo = &m_InstrStreams->m_PeArray;
        break;
    case EngineId::Activation:
        engInfo = &m_InstrStreams->m_ActEng;
        break;
    default:
        Assert(false, "Wrong EngineId for Semaphore instruction: ", static_cast<int>(engId));
    }
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

template<>
void WaveCode::writeInstruction<compisa::DmaTriggerInstr>(const compisa::DmaTriggerInstr& instruction, EngineId engId) const
{
    Assert(qBinFileRuntimeKelf(), "DmaTriggerInstr is available in RuntimeKelf binary only");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    InstrStreams::OneEngInfo* engInfo = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engInfo = &m_InstrStreams->m_PoolEng;
        break;
    case EngineId::PeArray:
        engInfo = &m_InstrStreams->m_PeArray;
        break;
    case EngineId::Activation:
        engInfo = &m_InstrStreams->m_ActEng;
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
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}


template<>
void WaveCode::writeInstruction<compisa::SimDmaCopyInstr>(const compisa::SimDmaCopyInstr& instruction, EngineId engId) const
{
    Assert(qBinFileSimKelf(), "SimDmaCopy is available in SimKelf binary only");
    instruction.CheckValidity();
    const kcc_int32 instSize = sizeof(instruction);

    InstrStreams::OneEngInfo* engInfo = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engInfo = &m_InstrStreams->m_PoolEng;
        break;
    case EngineId::PeArray:
        engInfo = &m_InstrStreams->m_PeArray;
        break;
    case EngineId::Activation:
        engInfo = &m_InstrStreams->m_ActEng;
        break;
    default:
        Assert(false, "Wrong EngineId for SimDmaCopy instruction: ", static_cast<int>(engId));
    }
    fwrite(&instruction, instSize, 1, engInfo->m_InstrStream);
    engInfo->m_Pc += instSize;
}

}}


