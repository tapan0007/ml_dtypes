
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wavecode/inc/wavecodesbatomfile.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtomFile::WaveCodeSbAtomFile(WaveCode* waveCode)
    : WaveCodeWaveOp(waveCode)
{}

void
WaveCodeSbAtomFile::generate(wave::WaveOp* waveOp)
{
    auto sbatomfileWaveOp = dynamic_cast<wave::SbAtomWaveOp*>(waveOp);
    assert(sbatomfileWaveOp);
}

}}

