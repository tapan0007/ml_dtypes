#include <map>


#include "nets/inc/network.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"

//#include "wavecode/inc/wavecodewaveop.hpp"
#include "wavecode/inc/wavecodesbatomfile.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"
#include "wavecode/inc/wavecodematmul.hpp"
#include "wavecode/inc/wavecodepool.hpp"
#include "wavecode/inc/wavecode.hpp"

namespace kcc {
namespace wavecode {

WaveCode::WaveCode(const nets::Network* network, const arch::Arch& arch)
    : m_Network(network)
    , m_Arch(arch)
{
    m_CodeMatMul     = std::make_unique<WaveCodeMatMul>(this);
    m_CodeSbAtomFile = std::make_unique<WaveCodeSbAtomFile>(this);
    m_CodeSbAtomSave = std::make_unique<WaveCodeSbAtomSave>(this);
    m_CodePool       = std::make_unique<WaveCodePool>(this);
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
    } else {
        assert(false && "WaveCode: Unsupported WaveOp");
    }
    return *m_CodeMatMul;
}

kcc_int64
WaveCode::getDramForNpyFile(const std::string& fileName)
{
    const auto it = m_NpyFile2DramAddress.find(fileName);
    if (it != m_NpyFile2DramAddress.end()) {
        return (*it).second;
    } else {
        return -1;
    }
}

void
WaveCode::recordDramForNpyFile(const std::string& fileName, kcc_int64 dramOffset)
{
    m_NpyFile2DramAddress[fileName] = dramOffset;
}

kcc_int64
WaveCode::gCurrentDramAddress(kcc_int64 sizeInBytes)
{
    const kcc_int64 currAddress = m_CurrentDramAddress;
    m_CurrentDramAddress += sizeInBytes;
    return currAddress;
}

}}

