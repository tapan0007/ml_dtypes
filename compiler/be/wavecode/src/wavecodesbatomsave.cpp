#include "tpb_isa_ldweights.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "layers/inc/layer.hpp"

#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtomSave::WaveCodeSbAtomSave(WaveCode* waveCode)
    : WaveCodeSbAtom(waveCode)
{}

void
WaveCodeSbAtomSave::generate(wave::WaveOp* waveOp)
{
    auto sbAtomSaveWaveOp = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveOp);
    assert(sbAtomSaveWaveOp);
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    kcc_int64 npyFileDramOffset = m_WaveCode->getDramForOutputNpyFile(sbAtomSaveWaveOp->gRefFileName());

    if (npyFileDramOffset < 0) {
        const kcc_int64 numPySize = sbAtomSaveWaveOp->gSaveDataSizeInBytes();
        npyFileDramOffset           = m_WaveCode->gCurrentDramAddress(numPySize);
        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomSaveWaveOp->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomSaveWaveOp->gRefFileShape();
        m_WaveCode->recordDramForOutputNpyFile(sbAtomSaveWaveOp->gRefFileName(), npyFileInfo);
    }

    const kcc_int64 numPartitions   = sbAtomSaveWaveOp->gOfmapCount();
    const kcc_int64 numBytesPerPart = sbAtomSaveWaveOp->gLength();
    const kcc_int64 addressInPart   = sbAtomSaveWaveOp->gAddressInPartition(0 /*offset in atom*/);
    const kcc_int64 stepSize = sbAtomSaveWaveOp->gPartitionStepBytes();

    SIM_MEMCPY statebufToDramInstr;
    statebufToDramInstr.nbytes       = numBytesPerPart;
    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
        statebufToDramInstr.src_address = stateBuf.gEntrySysAddress(partIdx, addressInPart);
        statebufToDramInstr.dst_address = npyFileDramOffset + sbAtomSaveWaveOp->gOffsetInFile() + (partIdx * stepSize);
        m_WaveCode->writeInstruction(statebufToDramInstr);
    }
}


}}


