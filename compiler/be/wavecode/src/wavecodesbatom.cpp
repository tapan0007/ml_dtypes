
#include "wave/inc/sbatomwaveop.hpp"
#include "wavecode/inc/wavecodesbatom.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtom::WaveCodeSbAtom(WaveCode* waveCode)
    : WaveCodeWaveOp(waveCode)
{}

void
WaveCodeSbAtom::generate(wave::WaveOp* waveOp)
{
    auto sbatomWaveOp = dynamic_cast<wave::SbAtomWaveOp*>(waveOp);
    assert(sbatomWaveOp);
}

}}

