#include "tpb_isa_ldweights.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/convlayer.hpp"

#include "wave/inc/sbatomfilewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomfile.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtomFile::WaveCodeSbAtomFile(WaveCode* waveCode)
    : WaveCodeSbAtom(waveCode)
{}

void
WaveCodeSbAtomFile::generate(wave::WaveOp* waveOp)
{
    const auto sbAtomFileWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(waveOp);
    assert(sbAtomFileWaveOp);
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    kcc_int64 npyFileDramOffset = m_WaveCode->getDramForInputNpyFile(sbAtomFileWaveOp->gRefFileName());
    if (npyFileDramOffset < 0) {
        SIM_WRNPY npyToDramInstr;
        // Load whole numpy file
        const kcc_int64 numPySize = sbAtomFileWaveOp->gLoadDataSizeInBytes();
        strcpy(npyToDramInstr.src_fname, sbAtomFileWaveOp->gRefFileName().c_str());
        npyFileDramOffset           = m_WaveCode->gCurrentDramAddress(numPySize);
        npyToDramInstr.dst_address  = npyFileDramOffset;
        m_WaveCode->recordDramForInputNpyFile(sbAtomFileWaveOp->gRefFileName(), npyFileDramOffset);
        m_WaveCode->writeInstruction(npyToDramInstr);
    }

    const kcc_int64 numPartitions   = sbAtomFileWaveOp->gIfmapCount();
    const kcc_int64 numBytesPerPart = sbAtomFileWaveOp->gLength();
    const kcc_int64 addressInPart   = sbAtomFileWaveOp->gAddressInPartition(0 /*offset in atom*/);
    const kcc_int64 stepSize = sbAtomFileWaveOp->gPartitionStepBytes();

    SIM_MEMCPY dramToStateBufInstr;
    dramToStateBufInstr.nbytes = numBytesPerPart;
    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
        dramToStateBufInstr.src_address = npyFileDramOffset + sbAtomFileWaveOp->gOffsetInFile() + (partIdx * stepSize);
        dramToStateBufInstr.dst_address = stateBuf.gEntrySysAddress(partIdx, addressInPart);
        m_WaveCode->writeInstruction(dramToStateBufInstr);
    }
}

}}

