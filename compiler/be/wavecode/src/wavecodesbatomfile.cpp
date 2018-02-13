#include "tpb_isa_ldweights.hpp"

#include "layers/inc/layer.hpp"
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
    auto sbatomfileWaveOp = dynamic_cast<wave::SbAtomWaveOp*>(waveOp);
    assert(sbatomfileWaveOp);
    layers::Layer* layer = sbatomfileWaveOp->gLayer();

    kcc_int64 dramOffset = m_WaveCode->getDramForNpyFile(sbatomfileWaveOp->gRefFileName());
    if (dramOffset < 0) {
        SIM_WRNPY npyToDramInstr;
        // Load whole numpy file
        strcpy(npyToDramInstr.src_fname, sbatomfileWaveOp->gRefFileName().c_str());

        kcc_int64 numPySize = layer->gDataType().gSizeInBytes();
        if (layers::ConvLayer* convLayer = dynamic_cast<layers::ConvLayer*>(layer)) {  // All Weights = CRSM
            numPySize *= convLayer->gNumIfmaps();    // C
            numPySize *= convLayer->gKernelHeight(); // R
            numPySize *= convLayer->gKernelWidth();  // S
            numPySize *= convLayer->gNumOfmaps();    // M
        } else {                                                       // All IFMAPs = NCHW
            // batching?                             // N
            numPySize *= convLayer->gNumIfmaps();    // C
            numPySize *= layer->gOfmapHeight();      // H
            numPySize *= layer->gOfmapWidth();       // W
        }
        dramOffset = m_WaveCode->gCurrentDramAddress(numPySize);
        npyToDramInstr.dst_address = dramOffset;
        m_WaveCode->recordDramForNpyFile(sbatomfileWaveOp->gRefFileName(), dramOffset);
        m_WaveCode->writeInstruction(npyToDramInstr, WaveCode::UseStream_StreamProc);
    }

    kcc_int64 stepSize = sbatomfileWaveOp->gDataType().gSizeInBytes();
    if (layers::ConvLayer* convLayer = dynamic_cast<layers::ConvLayer*>(layer)) {  // Weights: step in numpy/dram = RSM
        stepSize *= convLayer->gKernelHeight(); // R
        stepSize *= convLayer->gKernelWidth();  // S
        stepSize *= convLayer->gNumOfmaps();    // M
    } else {                                                       // IFMAPS: step in numpy/dram = NHW
        // batching?                             // N
        stepSize *= layer->gOfmapHeight();      // H
        stepSize *= layer->gOfmapWidth();       // W
    }

    const kcc_int64 numBytesPerPart = sbatomfileWaveOp->gLength();
    const kcc_int64 numPartitions = sbatomfileWaveOp->gNumPartitions();


    SIM_MEMCPY dramToStateBufInstr;
    dramToStateBufInstr.nbytes = numBytesPerPart;
    const kcc_int64 addressInPart = sbatomfileWaveOp->gAtomId() * layer->gNetwork()->gWaveAtomSize() + 
    for (int32 partIdx = 0; partIdx < numPartitions; ++partIdx) {
        dramToStateBufInstr.src_address = dramOffset + sbatomfileWaveOp->gOffsetInFile() + (partIdx * stepSize);

        dramToStateBufInstr.dst_address = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        m_WaveCode->writeInstruction(dramToStatBufInstr, WaveCode::UseStream_StreamProc);
    }
}

}}

