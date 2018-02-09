#include "nets/inc/network.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

//#include "wavecode/inc/wavecodewaveop.hpp"
#include "wavecode/inc/wavecodesbatomfile.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"
#include "wavecode/inc/wavecodematmul.hpp"
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
}

WaveCode::~WaveCode() = default;


void
WaveCode::generate(const char* objFileName)
{
    m_ObjFile = std::fopen(objFileName, "w");
    assert(m_ObjFile && "Object file is null");

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
    } else {
        assert(false && "WaveCode: Unsupported WaveOp");
    }
    return *m_CodeMatMul;
}

}}

