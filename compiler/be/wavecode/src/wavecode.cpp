#include <map>


//#include "shared/inc/uarch_cfg.hpp"


#include "compisa/inc/compisaldweights.hpp"
#include "compisa/inc/compisamatmul.hpp"

#include "compisa/inc/compisapool.hpp"
#include "compisa/inc/compisamatadd.hpp"

#include "compisa/inc/compisaactivation.hpp"

#include "compisa/inc/compisaset.hpp"
#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisawrite.hpp"

#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisasimwrnpy.hpp"
#include "compisa/inc/compisasimrdnpy.hpp"


#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"


#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/resaddwaveop.hpp"

//#include "wavecode/inc/wavecodewaveop.hpp"
#include "wavecode/inc/wavecodesbatomload.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"
#include "wavecode/inc/wavecodematmul.hpp"
#include "wavecode/inc/wavecodepool.hpp"
#include "wavecode/inc/wavecodeactivation.hpp"
#include "wavecode/inc/wavecoderesadd.hpp"

#include "wavecode/inc/wavecode.hpp"

namespace kcc {
namespace wavecode {

WaveCode::WaveCode(nets::Network* network, const arch::Arch& arch)
    : m_Network(network)
    , m_Arch(arch)
{
    m_CodeMatMul            = std::make_unique<WaveCodeMatMul>(*this);
    m_CodeSbAtomLoad        = std::make_unique<WaveCodeSbAtomLoad>(*this);
    m_CodeSbAtomSave        = std::make_unique<WaveCodeSbAtomSave>(*this);
    m_CodePool              = std::make_unique<WaveCodePool>(*this);
    m_CodeActivation        = std::make_unique<WaveCodeActivation>(*this);
    m_CodeResAdd            = std::make_unique<WaveCodeResAdd>(*this);

    m_CurrentDramAddress    = DDRC0_PORT0;
}

WaveCode::~WaveCode() = default;


void
WaveCode::generate(const InstrStreams& instrStreams, bool parallelStreams)
{
    m_ParallelStreams = parallelStreams;

    m_InstrStreams = &instrStreams;
    for (auto waveOp : m_Network->gWaveOps()) {
        auto& codeGen = getCodeGen(waveOp);
        codeGen.generate(waveOp);
    }
    saveAllNpyFiles();
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




template<>
void WaveCode::writeInstruction<compisa::MatMulInstr>(const compisa::MatMulInstr& instruction)
{
    checkForNoSync(instruction.sync);

    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PeArrayInstrStream);
}

template<>
void WaveCode::writeInstruction<compisa::LdWeightsInstr>(const compisa::LdWeightsInstr& instruction)
{
    checkForNoSync(instruction.sync);

    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PeArrayInstrStream);
}

template<>
void WaveCode::writeInstruction<compisa::PoolInstr>(const compisa::PoolInstr& instruction)
{
    checkForNoSync(instruction.sync);

    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PoolEngInstrStream);
}


template<>
void WaveCode::writeInstruction<compisa::ActivationInstr >(const compisa::ActivationInstr & instruction)
{
    checkForNoSync(instruction.sync);

    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_ActEngInstrStream);
}


template<>
void WaveCode::writeInstruction<compisa::MatAddInstr>(const compisa::MatAddInstr & instruction)
{
    checkForNoSync(instruction.sync);

    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PoolEngInstrStream);
}


template<>
void WaveCode::writeInstruction<compisa::SimRdNpyInstr>(const compisa::SimRdNpyInstr& instruction)
{
    checkForNoSync(instruction.sync);

    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_DmaInstrStream);
}

template<>
void WaveCode::writeInstruction<compisa::SimWrNpyInstr>(const compisa::SimWrNpyInstr& instruction)
{
    checkForNoSync(instruction.sync);

    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_DmaInstrStream);
}

template<>
void WaveCode::writeInstruction<compisa::SimMemCpyInstr>(const compisa::SimMemCpyInstr& instruction)
{
    checkForNoSync(instruction.sync);

    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_DmaInstrStream);
}





#if 0
template<>
void WaveCode::writeInstruction<compisa::WriteInstr>(const compisa::WriteInstr& instruction, EngineId engId)
{
    checkForNoSync(instruction.sync);

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
        Assert(false, "Wrong EngineId ", static_cast<int>(engId));
    }
}
#endif



template<>
void WaveCode::writeInstruction<compisa::WaitInstr>(const compisa::WaitInstr& instruction, EngineId engId)
{
    Assert(qParallelStreams(), "Cannot generate WAIT for event instruction in serial mode");

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
        Assert(false, "Wrong EngineId ", static_cast<int>(engId));
    }
}


template<>
void WaveCode::writeInstruction<compisa::SetInstr>(const compisa::SetInstr& instruction, EngineId engId)
{
    Assert(qParallelStreams(), "Cannot generate SET event instruction in serial mode");

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
        Assert(false, "Wrong EngineId ", static_cast<int>(engId));
    }
}










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
        dramToNpyInstr.sync.wait_event_id      = 0;
        dramToNpyInstr.sync.wait_event_mode    = eventWaitMode2Int(events::EventWaitMode::NoEvent);
        dramToNpyInstr.sync.set_event_id      = 0;
        dramToNpyInstr.sync.set_event_mode    = eventSetMode2Int(events::EventSetMode::NoEvent);

        strcpy(dramToNpyInstr.dst_fname, (*it).first.c_str());
        const NpyFileInfo& npyFileInfo((*it).second);
        dramToNpyInstr.src_address          = npyFileInfo.m_FileDramOffset;
        dramToNpyInstr.dst_ndims            = 4;
        for (int i = 0; i < dramToNpyInstr.dst_ndims; ++i) {
            dramToNpyInstr.dst_dims[i]   = npyFileInfo.m_RefFileShape[i];
        }
        dramToNpyInstr.dtype             = npyFileInfo.m_SimTypeId;

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
WaveCode::checkForNoSync(const TPB_CMD_SYNC& sync) const
{
    Assert(events::SET_EVENT_INVALID != sync.set_event_mode, "Invalid set event mode");
    Assert(events::WAIT_EVENT_INVALID != sync.wait_event_mode, "Invalid wait event mode");

    if (qParallelStreams()) {
        return;
    }
    Assert(NO_SET_EVENT == sync.set_event_mode,
        "Code generation: set event mode should be NONE in serial execution");

    Assert(NO_WAIT_EVENT == sync.wait_event_mode,
        "Code generation: wait event mode should be NONE in serial execution");
}





WaveCode::NpyFileInfo::NpyFileInfo()
{
    m_FileDramOffset    = -1;
    m_Dirty             = false;
    m_SimTypeId         = INVALID_ARBPRECTYPE;
}

}}

