#include "utils/inc/asserter.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "events/inc/events.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"

namespace kcc {
namespace wavecode {



//************************************************************************
WaveCodeSbAtomSave::WaveCodeSbAtomSave(WaveCodeRef waveCode)
    : WaveCodeSbAtom(waveCode)
{}




//************************************************************************
void
WaveCodeSbAtomSave::calcOutputSize(const wave::SbAtomSaveWaveOp* sbAtomSaveWaveop)
{
    const utils::DataType&    dtype(sbAtomSaveWaveop->gDataType());
    const utils::TensorParams::ShapeType& shape(sbAtomSaveWaveop->gRefFileShape ());
    kcc_int64 sz = dtype.gSizeInBytes();
    for (auto n : shape) {
        sz *= n;
    }
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    const kcc_int64 existingSz = kelfDma.gOutputSizeBytes(sbAtomSaveWaveop->gRefFileName());
    if (existingSz < 0) {
        kelfDma.rOutputSizeBytes(sz, sbAtomSaveWaveop->gRefFileName());
    } else {
        if (m_WaveCode.qBinFileRuntimeKelf()) {
            Assert(existingSz == sz,
                "Previously calculated output size ", existingSz,
                " != current size ", sz, " from AtomSave ",
                sbAtomSaveWaveop->gName());
        }
    }
}


//************************************************************************

}}


