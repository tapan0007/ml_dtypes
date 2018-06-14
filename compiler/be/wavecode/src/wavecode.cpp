#include <map>

#include "aws_tonga_isa_tpb_common.h"

#include "shared/inc/uarch_cfg.hpp"


#include "utils/inc/debug.hpp"

#include "compisa/inc/compisaldweights.hpp"
#include "compisa/inc/compisamatmul.hpp"

#include "compisa/inc/compisapool.hpp"

#include "compisa/inc/compisaactivate.hpp"

#include "compisa/inc/compisaset.hpp"
#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisaclear.hpp"
#include "compisa/inc/compisanop.hpp"

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
#include "compisa/inc/compisasimrdnpy.hpp"

#include "compisa/inc/compisadmatrigger.hpp"
#include "compisa/inc/compisasimdmacopy.hpp"


#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"


#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/resaddwaveop.hpp"
#include "wave/inc/barrierwaveop.hpp"
#include "wave/inc/nopwaveop.hpp"

//#include "wavecode/inc/wavecodewaveop.hpp"
#include "wavecode/inc/wavecodesbatomload.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"
#include "wavecode/inc/wavecodematmul.hpp"
#include "wavecode/inc/wavecodepool.hpp"
#include "wavecode/inc/wavecodeactivation.hpp"
#include "wavecode/inc/wavecoderesadd.hpp"
#include "wavecode/inc/wavecodebarrier.hpp"
#include "wavecode/inc/wavecodenop.hpp"

#include "wavecode/inc/wavecode.hpp"

namespace kcc {
namespace wavecode {

WaveCode::WaveCode(nets::Network* network, const arch::Arch& arch)
    : m_Network(network)
    , m_Arch(arch)
    , m_DmaDescription()
{
    m_CodeMatMul            = std::make_unique<WaveCodeMatMul>(*this);
    m_CodeSbAtomLoad        = std::make_unique<WaveCodeSbAtomLoad>(*this);
    m_CodeSbAtomSave        = std::make_unique<WaveCodeSbAtomSave>(*this);
    m_CodePool              = std::make_unique<WaveCodePool>(*this);
    m_CodeActivation        = std::make_unique<WaveCodeActivation>(*this);
    m_CodeResAdd            = std::make_unique<WaveCodeResAdd>(*this);
    m_CodeBarrier           = std::make_unique<WaveCodeBarrier>(*this);
    m_CodeNop               = std::make_unique<WaveCodeNop>(*this);

    m_CurrentDramAddress    = DDRC0_PORT0;
}

WaveCode::~WaveCode() = default;


void
WaveCode::determinePrecSbEdges()
{
    const std::array<EngineId, 3> engineIds = { {
        EngineId::Pooling,
        EngineId::Activation,
        EngineId::PeArray
    } };

    for (auto waveop : m_Network->gWaveOps()) {
        if (!waveop->qSbAtomWaveOp()) {
            continue;
        }
        if (waveop->gPrevWaveEdges().size() == 0) {
            continue; // initial loads
        }

        wave::WaveEdge* chosenPrevEdge = nullptr;
        for (auto engId : engineIds) {
            if (!chosenPrevEdge) {
                for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
                    if (prevWaveEdge->gFromOp()->gEngineId() == engId) {
                        chosenPrevEdge = prevWaveEdge;
                        break;
                    }
                }
                if (chosenPrevEdge) {
                    break;
                }
            }
        }
        // Chosen edge exist if on TPB engine.
        // If this is SbAtomSave --> SbAtomLoad, there will not be a chosen edge.
        if (chosenPrevEdge) {
            chosenPrevEdge->rChosenForSuccSbAtom(true);
        }
    }
}

void
WaveCode::generate(const InstrStreams& instrStreams, bool parallelStreams)
{
    m_ParallelStreams = parallelStreams;

    m_InstrStreams = &instrStreams;
    if (qGenerateKelf()) {
        determinePrecSbEdges();
    }
    for (auto waveOp : m_Network->gWaveOps()) {
        auto& codeGen = getCodeGen(waveOp);
        codeGen.generate(waveOp);
    }
    if (! qGenerateKelf()) {
        saveAllNpyFiles();
    }
    if (qBinFileRuntimeKelf()) {
        kelf::DmaDescription& dmaDescr(gDmaDescription());
        dmaDescr.writeDmaDescriptors(m_InstrStreams->m_PeArrayBinFile.c_str(), EngineId::PeArray);
        dmaDescr.writeDmaDescriptors(m_InstrStreams->m_ActEngBinFile.c_str(), EngineId::Activation);
        dmaDescr.writeDmaDescriptors(m_InstrStreams->m_PoolEngBinFile.c_str(), EngineId::Pooling);
        dmaDescr.writeInOutDescriptors();
        dmaDescr.writeDefinitions();
    }
}

WaveCodeWaveOp&
WaveCode::getCodeGen(const wave::WaveOp* waveOp)
{
    if (dynamic_cast<const wave::MatMulWaveOp*>(waveOp)) {
        return *m_CodeMatMul;
    } else if (dynamic_cast<const wave::SbAtomLoadWaveOp*>(waveOp)) {
        return *m_CodeSbAtomLoad;
    } else if (dynamic_cast<const wave::SbAtomSaveWaveOp*>(waveOp)) {
        return *m_CodeSbAtomSave;
    } else if (dynamic_cast<const wave::PoolWaveOp*>(waveOp)) {
        return *m_CodePool;
    } else if (dynamic_cast<const wave::ActivationWaveOp*>(waveOp)) {
        return *m_CodeActivation;
    } else if (dynamic_cast<const wave::ResAddWaveOp*>(waveOp)) {
        return *m_CodeResAdd;
    } else if (dynamic_cast<const wave::BarrierWaveOp*>(waveOp)) {
        return *m_CodeBarrier;
    } else if (dynamic_cast<const wave::NopWaveOp*>(waveOp)) {
        return *m_CodeNop;
    } else {
        assert(false && "WaveCode: Unsupported WaveOp");
    }
    return *m_CodeMatMul;
}

kcc_int64
WaveCode::getDramForNpyFile(const std::string& fileName)
{
    const auto it = m_NpyFile2DramAddress.find(fileName);
    if (m_NpyFile2DramAddress.end() != it) {
        return (*it).second.m_FileDramOffset;
    } else {
        return -1;
    }
}


void
WaveCode::recordDramForNpyFile(const std::string& fileName, const NpyFileInfo& npyFileInfo)
{
    m_NpyFile2DramAddress[fileName] = npyFileInfo;
}


kcc_int64
WaveCode::gCurrentDramAddress(kcc_int64 sizeInBytes)
{
    const kcc_int64 currAddress = m_CurrentDramAddress;
    m_CurrentDramAddress += sizeInBytes;
    return currAddress;
}



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
void WaveCode::writeInstruction<compisa::SetInstr>(const compisa::SetInstr& instruction, EngineId engId)
{
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








/***********************************************************************
 * Misc
***********************************************************************/
void
WaveCode::markDramDirty(const std::string& fileName)
{
    const auto it = m_NpyFile2DramAddress.find(fileName);
    assert(m_NpyFile2DramAddress.end() != it && "Setting dirty flag on non-existant file");
    (*it).second.m_Dirty = true;
}

void
WaveCode::saveAllNpyFiles ()
{

    const auto itE = m_NpyFile2DramAddress.cend();
    auto it = m_NpyFile2DramAddress.cbegin();
    for (; itE != it; ++it) {
        if (! (*it).second.m_Dirty) {
            continue;
        }
        compisa::SimRdNpyInstr dramToNpyInstr;
        dramToNpyInstr.inst_events.wait_event_idx    = 0;
        dramToNpyInstr.inst_events.wait_event_mode   = eventWaitMode2Isa(events::EventWaitMode::DontWait);
        dramToNpyInstr.inst_events.set_event_idx     = 0;
        dramToNpyInstr.inst_events.set_event_mode    = eventSetMode2Isa(events::EventSetMode::DontSet);

        strcpy(dramToNpyInstr.dst_fname, (*it).first.c_str());
        const NpyFileInfo& npyFileInfo((*it).second);
        dramToNpyInstr.src_addr         = npyFileInfo.m_FileDramOffset;
        dramToNpyInstr.dst_ndims        = 4;
        for (int i = 0; i < dramToNpyInstr.dst_ndims; ++i) {
            dramToNpyInstr.dst_dims[i]  = npyFileInfo.m_RefFileShape[i];
        }
        dramToNpyInstr.dtype            = npyFileInfo.m_SimTypeId;

        this->writeInstruction(dramToNpyInstr);
    }
}

kcc_uint64
WaveCode::calculateEventAddress(EngineId engId, events::EventId eventId) const
{
    const arch::Arch& arch(arch::Arch::gArch());

    switch (engId) {
    case EngineId::Pooling:
    case EngineId::PeArray:
    case EngineId::Activation:
        return arch.gTpbEventBase() + eventId; // 1 byte per eventId
        break;

    case EngineId::StreamProc:
        return arch.gSpEventBase()  + eventId;

    case EngineId::DmaEng:
        return arch.gTpbEventBase() + eventId;

    case EngineId::AnyEng:
        Assert(false, "AnyEng not allowed, engine id ", static_cast<int>(engId));

    case EngineId::None:
        Assert(false, "Bad engine id ", static_cast<int>(engId));
    }
    Assert(false, "Bad engine id ", static_cast<int>(engId));
    return 0;
}


void
WaveCode::checkForNoSync(const TONGA_ISA_TPB_INST_EVENTS& inst_events) const
{
    Assert(events::qEventSetModeValid(inst_events.set_event_mode), "Invalid set event mode");
    Assert(events::qEventWaitModeValid(inst_events.wait_event_mode), "Invalid wait event mode");

    if (qParallelStreams()) {
        return;
    }

    Assert(TONGA_ISA_TPB_MODE_SET_NONE == inst_events.set_event_mode,
        "Code generation: set event mode should be NONE in serial execution");

    Assert(TONGA_ISA_TPB_MODE_WAIT_NONE == inst_events.wait_event_mode,
        "Code generation: wait event mode should be NONE in serial execution");
}





WaveCode::NpyFileInfo::NpyFileInfo()
{
    m_FileDramOffset    = -1;
    m_Dirty             = false;
    m_SimTypeId         = TONGA_ISA_TPB_DTYPE_INVALID;
}



WaveCode::InstrStreams::~InstrStreams()
{
    closeAll();
}

void
WaveCode::InstrStreams::closeAll()
{
    if (m_StreamProcInstrStream) {
        fclose(m_StreamProcInstrStream);
        m_StreamProcInstrStream = nullptr;
    }
    if (m_PeArrayInstrStream) {
        fclose(m_PeArrayInstrStream);
        m_PeArrayInstrStream = nullptr;
    }
    if (m_PoolEngInstrStream) {
        fclose(m_PoolEngInstrStream);
        m_PoolEngInstrStream = nullptr;
    }
    if (m_ActEngInstrStream) {
        fclose(m_ActEngInstrStream);
        m_ActEngInstrStream = nullptr;
    }
    if (m_DmaInstrStream) {
        fclose(m_DmaInstrStream);
        m_DmaInstrStream = nullptr;
    }
}

}}

