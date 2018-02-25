
#include <sstream>


#include "utils/inc/datatype.hpp"

#include "layers/inc/layer.hpp"
#include "layers/inc/convlayer.hpp"
#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"

#include "wave/inc/sbatomfilewaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

SbAtomFileWaveOp::SbAtomFileWaveOp(
        const SbAtomFileWaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : SbAtomWaveOp(params, prevWaveOps)
    , m_IfmapCount(params.m_IfmapCount)
    , m_IfmapsFoldIdx(params.m_IfmapsFoldIdx)
    , m_IfmapsReplicate(params.m_IfmapsReplicate)
{
    assert(params.verify());
}

kcc_int64
SbAtomFileWaveOp::gLoadDataSizeInBytes () const
{
    const layers::Layer* layer = m_Layer;

    kcc_int64 numPySize = gDataType().gSizeInBytes();
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
    } else if (layer->qConstLayer()) {
        auto constLayer = dynamic_cast<const layers::ConstLayer*>(layer);  // All IFMAPs = NCHW
        assert(constLayer && "Const Layer expected");
        // batching?                             // N
        numPySize *= constLayer->gNumOfmaps();    // C
        numPySize *= constLayer->gOfmapHeight();  // H
        numPySize *= constLayer->gOfmapWidth();   // W
    } else {
        assert(false && "Conv or Input layer expected");
    }
    return numPySize;
}


bool
SbAtomFileWaveOp::verify() const
{
    if (! this->SbAtomWaveOp::verify()) {
        return false;
    }
    if (m_IfmapCount < 1) {
        return false;
    }
    if (m_IfmapsFoldIdx < 0) {
        return false;
    }
    // bool m_IfmapsReplicate
    return true;
}





bool
SbAtomFileWaveOp::Params::verify() const
{
    if (! this->SbAtomWaveOp::Params::verify()) {
        return false;
    }
    if (m_IfmapsFoldIdx < 0) {
        return false;
    }
    // bool m_IfmapsReplicate
    return true;
}


}}


