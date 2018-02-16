#include "tpb_isa_ldweights.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
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
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    const kcc_int64 numPartitions       = sbatomsaveWaveOp->gOfmapCount();
    const kcc_int64 numBytesPerPart     = sbatomsaveWaveOp->gLength();
    const kcc_int64 numPySize           = numPartitions * numBytesPerPart;
    const kcc_int64 addressInPart       = sbatomsaveWaveOp->gAtomId() * sbatomsaveWaveOp->gWaveAtomSize(); // offset = 0
    const kcc_int64 npyFileDramOffset   = m_WaveCode->gCurrentDramAddress(numPySize);
    const kcc_int64 stepSize            = numBytesPerPart;

    SIM_MEMCPY statebufToDramInstr;
    statebufToDramInstr.nbytes       = numBytesPerPart;
    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
        statebufToDramInstr.src_address = stateBuf.gEntryTpbAddress(partIdx, addressInPart); 
        statebufToDramInstr.dst_address = npyFileDramOffset + sbatomsaveWaveOp->gOffsetInFile() + (partIdx * stepSize);

        m_WaveCode->writeInstruction(statebufToDramInstr, WaveCode::UseStream_StreamProc);
    }




    SIM_RDNPY dramToNpyInstr;
    dramToNpyInstr.src_address       = statebufToDramInstr.dst_address;
    strcpy(dramToNpyInstr.dst_fname, sbatomsaveWaveOp->gRefFileName().c_str());
    dramToNpyInstr.dst_ndims         = 4;
    for (int i = 0; i < dramToNpyInstr.dst_ndims; ++i) {
        dramToNpyInstr.dst_dims[i]   = sbatomsaveWaveOp->gRefFileShape()[i];
    }
    dramToNpyInstr.dtype             = waveOp->gDataType().gSimTypeId();

    m_WaveCode->writeInstruction(dramToNpyInstr, WaveCode::UseStream_StreamProc);
}

}}


