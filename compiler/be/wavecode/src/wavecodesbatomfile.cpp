#include "tpb_isa_ldweights.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

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
    const layers::Layer* layer = sbatomfileWaveOp->gLayer();
    const arch::StateBuffer& stateBuf(layer->gArch().gStateBuffer());

    kcc_int64 dramOffset = m_WaveCode->getDramForNpyFile(sbatomfileWaveOp->gRefFileName());
    if (dramOffset < 0) {
        SIM_WRNPY npyToDramInstr;
        // Load whole numpy file
        strcpy(npyToDramInstr.src_fname, sbatomfileWaveOp->gRefFileName().c_str());

        kcc_int64 numPySize = layer->gDataType().gSizeInBytes();
        if (auto convLayer = dynamic_cast<const layers::ConvLayer*>(layer)) {  // All Weights = CRSM
            layers::Layer* prevLayer = convLayer->gPrevLayer(0);
            numPySize *= prevLayer->gNumOfmaps();    // C
            numPySize *= convLayer->gKernelHeight(); // R
            numPySize *= convLayer->gKernelWidth();  // S
            numPySize *= convLayer->gNumOfmaps();    // M
        } else {                                                       // All IFMAPs = NCHW
            // batching?                             // N
            numPySize *= convLayer->gNumOfmaps();    // C
            numPySize *= layer->gOfmapHeight();      // H
            numPySize *= layer->gOfmapWidth();       // W
        }
        dramOffset = m_WaveCode->gCurrentDramAddress(numPySize);
        npyToDramInstr.dst_address = dramOffset;
        m_WaveCode->recordDramForNpyFile(sbatomfileWaveOp->gRefFileName(), dramOffset);
        m_WaveCode->writeInstruction(npyToDramInstr, WaveCode::UseStream_StreamProc);
    }

    kcc_int64 stepSize = sbatomfileWaveOp->gDataType().gSizeInBytes();
    if (auto convLayer = dynamic_cast<const layers::ConvLayer*>(layer)) {  // Weights: step in numpy/dram = RSM
        stepSize *= convLayer->gKernelHeight(); // R
        stepSize *= convLayer->gKernelWidth();  // S
        stepSize *= convLayer->gNumOfmaps();    // M
    } else {                                                       // IFMAPS: step in numpy/dram = NHW
        // batching?                             // N
        stepSize *= layer->gOfmapHeight();      // H
        stepSize *= layer->gOfmapWidth();       // W
    }

    const kcc_int64 numBytesPerPart = sbatomfileWaveOp->gLength();
    const kcc_int64 numPartitions = sbatomfileWaveOp->gIfmapCount();


    SIM_MEMCPY dramToStateBufInstr;
    dramToStateBufInstr.nbytes = numBytesPerPart;
    const kcc_int64 addressInPart = sbatomfileWaveOp->gAtomId() * layer->gWaveAtomSize();
    for (kcc_int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
        dramToStateBufInstr.src_address = dramOffset + sbatomfileWaveOp->gOffsetInFile() + (partIdx * stepSize);

        dramToStateBufInstr.dst_address = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        m_WaveCode->writeInstruction(dramToStateBufInstr, WaveCode::UseStream_StreamProc);
    }
}

}}

