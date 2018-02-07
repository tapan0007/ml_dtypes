#include "nets/inc/network.hpp"

#include "wavecode/inc/wavecode.hpp"

namespace kcc {
namespace wavecode {

WaveCode::WaveCode(const nets::Network* network, const arch::Arch& arch)
    : m_Network(network)
    , m_Arch(arch)
{}


void
WaveCode::generate(const char* /*fileName*/)
{
    for (auto waveOp : m_Network->gWaveOps()) {
        auto& codeGen = getCodeGen(waveOp);
        codeGen.generate(waveOp);
    }
}

}}

