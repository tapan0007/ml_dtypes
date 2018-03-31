

#include "compisa/inc/compisamatadd.hpp"



#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/resaddwaveop.hpp"
#include "wave/inc/barrierwaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodebarrier.hpp"

namespace kcc {
namespace wavecode {

WaveCodeBarrier::WaveCodeBarrier(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeBarrier::generate(wave::WaveOp* waveOp)
{
    auto barrierWaveop = dynamic_cast<wave::BarrierWaveOp*>(waveOp);
    Assert(barrierWaveop, "Codegen expects barrier waveop");

    //************************************************************************
    if (qParallelStreams()) {
        processIncomingEdges(barrierWaveop);
        processOutgoingEdges(barrierWaveop);
    }
}


}}


