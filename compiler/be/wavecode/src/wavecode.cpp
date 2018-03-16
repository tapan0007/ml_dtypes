#include <map>


#include "shared/inc/uarch_cfg.hpp"
#include "tcc/inc/tcc.hpp"


#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"

#include "events/inc/eventmgr.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/resaddwaveop.hpp"

//#include "wavecode/inc/wavecodewaveop.hpp"
#include "wavecode/inc/wavecodesbatomfile.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"
#include "wavecode/inc/wavecodematmul.hpp"
#include "wavecode/inc/wavecodepool.hpp"
#include "wavecode/inc/wavecodeactivation.hpp"
#include "wavecode/inc/wavecoderesadd.hpp"

#include "wavecode/inc/wavecode.hpp"

namespace kcc {
namespace wavecode {

WaveCode::WaveCode(const nets::Network* network, const arch::Arch& arch)
    : m_Network(network)
    , m_Arch(arch)
{
    m_CodeMatMul            = std::make_unique<WaveCodeMatMul>(this);
    m_CodeSbAtomFile        = std::make_unique<WaveCodeSbAtomFile>(this);
    m_CodeSbAtomSave        = std::make_unique<WaveCodeSbAtomSave>(this);
    m_CodePool              = std::make_unique<WaveCodePool>(this);
    m_CodeActivation        = std::make_unique<WaveCodeActivation>(this);
    m_CodeResAdd            = std::make_unique<WaveCodeResAdd>(this);

    m_CurrentDramAddress    = DDRC0_PORT0;
}

WaveCode::~WaveCode() = default;


void
WaveCode::generate(const InstrStreams& instrStreams, bool parallelStreams)
{
    if (parallelStreams) {
        events::EventMgr eventMgr(*m_Network);
        eventMgr.processWaveops();
    }

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
    } else if (dynamic_cast<const wave::SbAtomFileWaveOp*>(waveOp)) {
        return *m_CodeSbAtomFile;
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
void WaveCode::writeInstruction<MATMUL>(const MATMUL& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PeArrayInstrStream);
}

template<>
void WaveCode::writeInstruction<LDWEIGHTS>(const LDWEIGHTS& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PeArrayInstrStream);
}

template<>
void WaveCode::writeInstruction<POOL>(const POOL& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PoolEngInstrStream);
}


template<>
void WaveCode::writeInstruction<ACTIVATION >(const ACTIVATION & instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_ActEngInstrStream);
}


template<>
void WaveCode::writeInstruction<MATADD>(const MATADD & instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PoolEngInstrStream);
}


template<>
void WaveCode::writeInstruction<SIM_RDNPY>(const SIM_RDNPY& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_DmaInstrStream);
}

template<>
void WaveCode::writeInstruction<SIM_WRNPY>(const SIM_WRNPY& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_DmaInstrStream);
}

template<>
void WaveCode::writeInstruction<SIM_MEMCPY>(const SIM_MEMCPY& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_DmaInstrStream);
}





#if 0
template<>
void WaveCode::writeInstruction<WRITE>(const WRITE& instruction, EngineId engId)
{
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
void WaveCode::writeInstruction<WAIT>(const WAIT& instruction, EngineId engId)
{
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
void WaveCode::writeInstruction<SET>(const SET& instruction, EngineId engId)
{
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
        SIM_RDNPY dramToNpyInstr;
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
WaveCode::calculateEventAddress(EngineId engId, EventId eventId) const
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

    case EngineId::None:
        Assert(false, "Bad engine id ", static_cast<int>(engId));
    }
    Assert(false, "Bad engine id ", static_cast<int>(engId));
    return 0;
}

}}

