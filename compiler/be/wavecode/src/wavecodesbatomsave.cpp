
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtomSave::WaveCodeSbAtomSave(WaveCode* waveCode)
    : WaveCodeWaveOp(waveCode)
{}

void
WaveCodeSbAtomSave::generate(wave::WaveOp* waveOp)
{
    auto sbatomsaveWaveOp = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveOp);
    assert(sbatomsaveWaveOp);
}

}}


