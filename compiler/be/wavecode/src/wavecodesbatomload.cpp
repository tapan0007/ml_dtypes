#include "utils/inc/debug.hpp"
#include "utils/inc/misc.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"


#include "events/inc/events.hpp"



#include "wave/inc/waveedge.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomload.hpp"

namespace kcc {
namespace wavecode {




//************************************************************************
WaveCodeSbAtomLoad::WaveCodeSbAtomLoad(WaveCodeRef waveCode)
    : WaveCodeSbAtom(waveCode)
{}




/***********************************************************************
***********************************************************************/
void
WaveCodeSbAtomLoad::calcInputSize(const wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    const std::string& refFile(sbAtomLoadWaveop->gRefFileName());
    if (sbAtomLoadWaveop->qContainWeights()) { // ifmap
        return;
    }
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    kelfDma.recordInFile(sbAtomLoadWaveop->gRefFileName());
    const utils::DataType&    dtype(sbAtomLoadWaveop->gDataType());
    const utils::TensorParams::ShapeType& shape(sbAtomLoadWaveop->gRefFileShape ());
    kcc_int64 sz = dtype.gSizeInBytes();
    for (auto n : shape) {
        sz *= n;
    }
    kelfDma.rInputSizeBytes(sz, refFile);
}

}}

