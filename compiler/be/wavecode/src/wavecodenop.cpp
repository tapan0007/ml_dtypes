

//#include "compisa/inc/compisamatadd.hpp"



#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/resaddwaveop.hpp"
#include "wave/inc/nopwaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodenop.hpp"

namespace kcc {
namespace wavecode {

WaveCodeNop::WaveCodeNop(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeNop::generate(wave::WaveOp* waveOp)
{
    auto nopWaveop = dynamic_cast<wave::NopWaveOp*>(waveOp);
    Assert(nopWaveop, "Codegen expects nop waveop");

    //************************************************************************
    if (qParallelStreams()) {
        processIncomingEdges(nopWaveop);
        processOutgoingEdges(nopWaveop);
    }
}


}}



