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
    : WaveCodeWaveOp(waveCode)
{}

void
WaveCodeSbAtomSave::generate(wave::WaveOp* waveOp)
{
    auto sbatomsaveWaveOp = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveOp);
    assert(sbatomsaveWaveOp);
    const layers::Layer* layer = sbatomsaveWaveOp->gLayer();

    kcc_int64 npyFileDramOffset = m_WaveCode->getDramForOutputNpyFile(sbatomsaveWaveOp->gRefFileName());

    if (npyFileDramOffset < 0) {
        kcc_int64 numPySize = layer->gDataType().gSizeInBytes();
        numPySize *= layer->gNumOfmaps();    // C
        numPySize *= layer->gOfmapHeight();  // H
        numPySize *= layer->gOfmapWidth();   // W
        npyFileDramOffset           = m_WaveCode->gCurrentDramAddress(numPySize);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = waveOp->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbatomsaveWaveOp->gRefFileShape();
        m_WaveCode->recordDramForOutputNpyFile(sbatomsaveWaveOp->gRefFileName(), npyFileInfo);
    }

    const kcc_int64 numPartitions       = sbatomsaveWaveOp->gOfmapCount();
    const kcc_int64 addressInPart       = sbatomsaveWaveOp->gAtomId() * sbatomsaveWaveOp->gWaveAtomSize(); // offset = 0
    const kcc_int64 numBytesPerPart     = sbatomsaveWaveOp->gLength();
    kcc_int64 stepSize                  = layer->gOfmapHeight();
    stepSize                            *= layer->gOfmapWidth();
    stepSize                            *= sbatomsaveWaveOp->gDataType().gSizeInBytes();

    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    SIM_MEMCPY statebufToDramInstr;
    statebufToDramInstr.nbytes       = numBytesPerPart;
    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
        statebufToDramInstr.src_address = stateBuf.gEntryTpbAddress(partIdx, addressInPart); 
        statebufToDramInstr.dst_address = npyFileDramOffset + sbatomsaveWaveOp->gOffsetInFile() + (partIdx * stepSize);

        m_WaveCode->writeInstruction(statebufToDramInstr);
    }
}


}}


