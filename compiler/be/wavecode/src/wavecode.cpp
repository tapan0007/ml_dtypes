#include "nets/inc/network.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomwaveop.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"
#include "wavecode/inc/wavecodematmul.hpp"
#include "wavecode/inc/wavecodesbatom.hpp"
#include "wavecode/inc/wavecode.hpp"

namespace kcc {
namespace wavecode {

WaveCode::WaveCode(const nets::Network* network, const arch::Arch& arch)
    : m_Network(network)
    , m_Arch(arch)
{
    m_CodeMatMul = std::make_unique<WaveCodeMatMul>(this);
    m_CodeSbAtom = std::make_unique<WaveCodeSbAtom>(this);
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
    } else if (dynamic_cast<const wave::SbAtomWaveOp*>(waveOp)) {
        return *m_CodeSbAtom;
    } else {
        assert(false && "WaveCode: Unsupported WaveOp");
    }
    return *m_CodeMatMul;
}

}}

