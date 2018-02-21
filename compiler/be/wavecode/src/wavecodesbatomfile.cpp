#include "tpb_isa_ldweights.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "layers/inc/inputlayer.hpp"
#include "layers/inc/convlayer.hpp"

#include "wave/inc/sbatomfilewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomfile.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtomFile::WaveCodeSbAtomFile(WaveCode* waveCode)
    : WaveCodeWaveOp(waveCode)
{}

void
WaveCodeSbAtomFile::generate(wave::WaveOp* waveOp)
{
    const auto sbatomfileWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(waveOp);
    assert(sbatomfileWaveOp);

    kcc_int64 npyFileDramOffset = m_WaveCode->getDramForInputNpyFile(sbatomfileWaveOp->gRefFileName());
    if (npyFileDramOffset < 0) {
        const layers::Layer* layer = sbatomfileWaveOp->gLayer();
        SIM_WRNPY npyToDramInstr;
        // Load whole numpy file
        strcpy(npyToDramInstr.src_fname, sbatomfileWaveOp->gRefFileName().c_str());

        kcc_int64 numPySize = layer->gDataType().gSizeInBytes();
        if (layer->qConvLayer()) {
            auto convLayer = dynamic_cast<const layers::ConvLayer*>(layer);  // All Weights = CRSM
            assert(convLayer && "Conv Layer expected");
            layers::Layer* prevLayer = convLayer->gPrevLayer(0);
            numPySize *= prevLayer->gNumOfmaps();    // C
            numPySize *= convLayer->gKernelHeight(); // R
            numPySize *= convLayer->gKernelWidth();  // S
            numPySize *= convLayer->gNumOfmaps();    // M
        } else if (layer->qInputLayer()) {
            auto inputLayer = dynamic_cast<const layers::InputLayer*>(layer);  // All IFMAPs = NCHW
            assert(inputLayer && "Input Layer expected");
            // batching?                             // N
            numPySize *= inputLayer->gNumOfmaps();    // C
            numPySize *= inputLayer->gOfmapHeight();  // H
            numPySize *= inputLayer->gOfmapWidth();   // W
        } else {
            assert(false && "Conv or Input layer expected");
        }
        npyFileDramOffset           = m_WaveCode->gCurrentDramAddress(numPySize);
        npyToDramInstr.dst_address  = npyFileDramOffset;
        m_WaveCode->recordDramForInputNpyFile(sbatomfileWaveOp->gRefFileName(), npyFileDramOffset);
        m_WaveCode->writeInstruction(npyToDramInstr);
    }

    // These are waveop params, not layer params
    const kcc_int64 numBytesPerPart = sbatomfileWaveOp->gLength();
    const kcc_int64 numPartitions   = sbatomfileWaveOp->gIfmapCount();
    const kcc_int64 addressInPart   = sbatomfileWaveOp->gAtomId() * sbatomfileWaveOp->gWaveAtomSize();

    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    SIM_MEMCPY dramToStateBufInstr;
    dramToStateBufInstr.nbytes = numBytesPerPart;
    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
        dramToStateBufInstr.src_address = npyFileDramOffset + sbatomfileWaveOp->gOffsetInFile() + (partIdx * numBytesPerPart);
        dramToStateBufInstr.dst_address = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        m_WaveCode->writeInstruction(dramToStateBufInstr);
    }
}

}}

