#include <map>


#include "uarch_cfg.hpp"
#include "tpb_isa_ldweights.hpp"
#include "tcc.hpp"

#include "nets/inc/network.hpp"

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
WaveCode::generate(const InstrStreams& instrStreams)
{
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
WaveCode::getDramForInputNpyFile(const std::string& fileName)
{
    const auto it = m_InputNpyFile2DramAddress.find(fileName);
    if (m_InputNpyFile2DramAddress.end() != it) {
        return (*it).second;
    } else {
        return -1;
    }
}

kcc_int64
WaveCode::getDramForOutputNpyFile(const std::string& fileName)
{
    const auto it = m_OutputNpyFile2DramAddress.find(fileName);
    if (it != m_OutputNpyFile2DramAddress.end()) {
        return (*it).second.m_FileDramOffset;
    } else {
        return -1;
    }
}

void
WaveCode::recordDramForInputNpyFile(const std::string& fileName, kcc_int64 dramOffset)
{
    m_InputNpyFile2DramAddress[fileName] = dramOffset;
}

void
WaveCode::recordDramForOutputNpyFile(const std::string& fileName, const NpyFileInfo& npyFileInfo)
{
    m_OutputNpyFile2DramAddress[fileName] = npyFileInfo;
}

kcc_int64
WaveCode::gCurrentDramAddress(kcc_int64 sizeInBytes)
{
    const kcc_int64 currAddress = m_CurrentDramAddress;
    m_CurrentDramAddress += sizeInBytes;
    return currAddress;
}



template<>
void WaveCode::writeInstruction<MATMUL>(MATMUL& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PeArrayInstrStream);
}

template<>
void WaveCode::writeInstruction<LDWEIGHTS>(LDWEIGHTS& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PeArrayInstrStream);
}

template<>
void WaveCode::writeInstruction<POOL>(POOL& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PoolEngInstrStream);
}


template<>
void WaveCode::writeInstruction<ACTIVATION >(ACTIVATION & instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_ActEngInstrStream);
}


template<>
void WaveCode::writeInstruction<MATADD>(MATADD & instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_PoolEngInstrStream);
}




template<>
void WaveCode::writeInstruction<SIM_RDNPY>(SIM_RDNPY& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_StreamProcInstrStream);
}

template<>
void WaveCode::writeInstruction<SIM_WRNPY>(SIM_WRNPY& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_StreamProcInstrStream);
}

template<>
void WaveCode::writeInstruction<SIM_MEMCPY>(SIM_MEMCPY& instruction)
{
    fwrite(&instruction, sizeof(instruction), 1, m_InstrStreams->m_StreamProcInstrStream);
}


void
WaveCode::saveAllNpyFiles ()
{

    const auto itE = m_OutputNpyFile2DramAddress.cend();
    auto it = m_OutputNpyFile2DramAddress.cbegin();
    for (; itE != it; ++it) {
        SIM_RDNPY dramToNpyInstr;
        strcpy(dramToNpyInstr.dst_fname, (*it).first.c_str());
        const NpyFileInfo& npyFileInfo((*it).second);
        dramToNpyInstr.src_address          = (*it).second.m_FileDramOffset;
        dramToNpyInstr.dst_ndims            = 4;
        for (int i = 0; i < dramToNpyInstr.dst_ndims; ++i) {
            dramToNpyInstr.dst_dims[i]   = npyFileInfo.m_RefFileShape[i];
        }
        dramToNpyInstr.dtype             = npyFileInfo.m_SimTypeId;

        this->writeInstruction(dramToNpyInstr);
    }
}

}}

