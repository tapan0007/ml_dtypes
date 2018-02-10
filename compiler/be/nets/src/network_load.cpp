#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>


#include "arch/inc/arch.hpp"

#include "layers/inc/layer.hpp"
#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/convlayer.hpp"
#include "layers/inc/relulayer.hpp"
#include "layers/inc/tanhlayer.hpp"
#include "layers/inc/maxpoollayer.hpp"
#include "layers/inc/avgpoollayer.hpp"
#include "layers/inc/resaddlayer.hpp"
#include "layers/inc/biasaddlayer.hpp"

#include "nets/inc/network.hpp"
#include "serialize/inc/serlayer.hpp"
#include "serialize/inc/serwaveop.hpp"

namespace kcc {

/*
namespace wave {
    class SbAtomFileWaveOp;
    class SbAtomSaveWaveOp;
    class MatMulWaveOp;
}
*/

namespace nets {

//--------------------------------------------------------
//--------------------------------------------------------
template<>
void
Network::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    archive(cereal::make_nvp(NetKey_NetName, m_Name));
    std::string dataType;
    archive(cereal::make_nvp(NetKey_DataType, dataType));

    if (dataType == DataTypeUint8::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeUint8>();

    } else if (dataType == DataTypeUint16::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeUint16>();

    } else if (dataType == DataTypeFloat16::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeFloat16>();

    } else if (dataType == DataTypeFloat32::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeFloat32>();

    } else {
        assert(0 && "Unsupported data type");
    }

    std::vector<serialize::SerLayer> serLayers;
    archive(cereal::make_nvp(NetKey_Layers, serLayers));
    kcc::utils::breakFunc(333);
    for (unsigned i = 0; i < serLayers.size(); ++i) {
        const serialize::SerLayer& serLayer(serLayers[i]);
        layers::Layer::Params params;
        params.m_LayerName = serLayer.gName();
        params.m_BatchFactor = serLayer.gBatchFactor();
        params.m_Network = this;
        params.m_RefFile = serLayer.gRefFile();
        params.m_RefFileFormat = serLayer.gOfmapFormat();

        FmapDesc fmap_desc(serLayer.gNumOfmaps(), serLayer.gOfmapHeight(), serLayer.gOfmapWidth());
        layers::Layer* layer = nullptr;
        if (serLayer.gTypeStr() == LayerTypeStr_Input) {
            assert(serLayer.gNumPrevLayers() == 0 && "Input layer should have zero inputs");
            layer = new layers::InputLayer(params, fmap_desc);
        } else if (serLayer.gTypeStr() == LayerTypeStr_Const) {
            assert(serLayer.gNumPrevLayers() == 0 && "Const layer should have zero inputs");
            layer = new layers::ConstLayer(params, fmap_desc);
        } else if (serLayer.gTypeStr() == LayerTypeStr_Conv) {
            assert(serLayer.gNumPrevLayers() == 1 && "Convolution layer should have one input");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Convolution: Unknown input layer");
            std::tuple<kcc_int32,kcc_int32> stride = std::make_tuple(
                                            serLayer.gStrideVertical(),
                                            serLayer.gStrideHorizontal());
            std::tuple<kcc_int32,kcc_int32> kernel = std::make_tuple(
                                            serLayer.gConvFilterHeight(),
                                            serLayer.gConvFilterWidth());
            std::tuple<kcc_int32,kcc_int32, kcc_int32,kcc_int32> padding = std::make_tuple(
                                            serLayer.gPaddingTop(),
                                            serLayer.gPaddingBottom(),
                                            serLayer.gPaddingLeft(),
                                            serLayer.gPaddingRight());

            const std::string filterFileName = serLayer.gKernelFile();
            const std::string filterTensorDimSemantics = serLayer.gKernelFormat();
            layers::ConvLayer::Params convParams(params);
            /*
            convParams.m_BatchingInWave    = serLayer.gBatchingInWave();
            */

            layer = new layers::ConvLayer(convParams, prevLayer,
                                          fmap_desc,
                                          stride, kernel, padding,
                                          filterFileName.c_str(),
                                          filterTensorDimSemantics.c_str());

        } else if (serLayer.gTypeStr() == LayerTypeStr_MaxPool || serLayer.gTypeStr() == LayerTypeStr_AvgPool) {
            assert(serLayer.gNumPrevLayers() == 1 && "Pool layer: number of inputs not 1");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Pool: Unknown previous layer");
            std::tuple<kcc_int32,kcc_int32> stride = std::make_tuple(
                                            serLayer.gStrideVertical(),
                                            serLayer.gStrideHorizontal());
            std::tuple<kcc_int32,kcc_int32> kernel = std::make_tuple(
                                            serLayer.gPoolKernelHeight(),
                                            serLayer.gPoolKernelWidth());
            std::tuple<kcc_int32,kcc_int32, kcc_int32,kcc_int32> padding = std::make_tuple(
                                            serLayer.gPaddingTop(),
                                            serLayer.gPaddingBottom(),
                                            serLayer.gPaddingLeft(),
                                            serLayer.gPaddingRight());

            if (serLayer.gTypeStr() == LayerTypeStr_MaxPool) {
                layer = new layers::MaxPoolLayer(
                        params,
                        prevLayer,
                        fmap_desc,
                        stride,
                        kernel,
                        padding);
            } else {
                layer = new layers::AvgPoolLayer(
                        params,
                        prevLayer,
                        fmap_desc,
                        stride,
                        kernel,
                        padding);
            }

        } else if (serLayer.gTypeStr() == LayerTypeStr_Relu) {
            assert(serLayer.gNumPrevLayers() == 1 && "Relu layer: number of inputs not 1");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Relu: Unknown previous layer");
            layer = new layers::ReluLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == LayerTypeStr_Tanh) {
            assert(serLayer.gNumPrevLayers() == 1 && "Tanh layer: number of inputs not 1");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Tanh: Unknown previous layer");
            layer = new layers::TanhLayer(params, prevLayer);

        } else if (serLayer.gTypeStr() == LayerTypeStr_ResAdd) {
            // TODO: check dimensions and types of inputs
            assert(serLayer.gNumPrevLayers() == 2 && "ResAdd layer should have two inputs");
            std::vector<layers::Layer*> prevLayers;
            for (const auto& prevLayerName : serLayer.gPrevLayers()) {
                layers::Layer* prevLayer = findLayer(prevLayerName);
                assert(prevLayer != nullptr && "RadAdd: Unknown previous layer");
                prevLayers.push_back(prevLayer);
            }
            layer = new layers::ResAddLayer(params, fmap_desc,prevLayers);
        } else if (serLayer.gTypeStr() == LayerTypeStr_BiasAdd) {
            // TODO check dimensions and types of inputs
            assert(serLayer.gNumPrevLayers() == 2 && "BiasAdd layer should have two inputs");
            std::vector<layers::Layer*> prevLayers;
            for (const auto& prevLayerName : serLayer.gPrevLayers()) {
                layers::Layer* prevLayer = findLayer(prevLayerName);
                assert(prevLayer != nullptr && "RadAdd: Unknown previous layer");
                prevLayers.push_back(prevLayer);
            }
            layer = new layers::BiasAddLayer(params, fmap_desc, prevLayers);
        } else {
            assert(false && "Unsuported layer");
        }

        assert(m_Name2Layer.find(params.m_LayerName) == m_Name2Layer.end());
        m_Name2Layer[params.m_LayerName] = layer;
    }
    assert(m_Layers.size() == serLayers.size() && "Layer mismatch count after input deserialization" );


    //===========================================================================
    if (m_UseWave) {
        std::vector<serialize::SerWaveOp> serWaveOps;
        archive(cereal::make_nvp(NetKey_WaveOps, serWaveOps));
        for (unsigned i = 0; i < serWaveOps.size(); ++i) {
            const serialize::SerWaveOp& serWaveOp(serWaveOps[i]);
    
            std::vector<wave::WaveOp*> prevWaveOps;
            wave::WaveOp* waveOp = nullptr;
    
            auto fillWaveOpParams = [this, &prevWaveOps](
                                        const serialize::SerWaveOp& serWaveOp,
                                        wave::WaveOp::Params& waveOpParams) -> void
            {
                waveOpParams.m_WaveOpName   = serWaveOp.gWaveOpName();
                waveOpParams.m_Layer        = this->findLayer(serWaveOp.gLayerName());
                assert(waveOpParams.m_Layer);
                for (const auto& prevWaveOpName : serWaveOp.gPreviousWaveOps()) {
                    prevWaveOps.push_back(findWaveOp(prevWaveOpName));
                }
            };
    
    
            if (serWaveOp.gWaveOpType() == wave::SbAtomFileWaveOp::gTypeStr()) {
                wave::SbAtomFileWaveOp::Params sbatomfileParams;
                fillWaveOpParams(serWaveOp, sbatomfileParams);
                sbatomfileParams.m_RefFileName       = serWaveOp.gRefFile();
                sbatomfileParams.m_BatchFoldIdx      = serWaveOp.gBatchFoldIdx();
                sbatomfileParams.m_AtomId            = serWaveOp.gAtomId();
                sbatomfileParams.m_Length            = serWaveOp.gLength();
                sbatomfileParams.m_OffsetInFile      = serWaveOp.gOffsetInFile();
    
                sbatomfileParams.m_IfmapsFoldIdx     = serWaveOp.gIfmapsFoldIdx();
                sbatomfileParams.m_IfmapsReplicate   = serWaveOp.qIfmapsReplicate();
    
                waveOp = new wave::SbAtomFileWaveOp(sbatomfileParams, prevWaveOps);
                assert(waveOp->gName() == sbatomfileParams.m_WaveOpName);
    
            } else if (serWaveOp.gWaveOpType() == wave::SbAtomSaveWaveOp::gTypeStr()) {
                wave::SbAtomSaveWaveOp::Params sbatomsaveParams;
                fillWaveOpParams(serWaveOp, sbatomsaveParams);
                sbatomsaveParams.m_RefFileName       = serWaveOp.gRefFile();
                sbatomsaveParams.m_BatchFoldIdx      = serWaveOp.gBatchFoldIdx();
                sbatomsaveParams.m_AtomId            = serWaveOp.gAtomId();
                sbatomsaveParams.m_Length            = serWaveOp.gLength();
                sbatomsaveParams.m_OffsetInFile      = serWaveOp.gOffsetInFile();
    
                sbatomsaveParams.m_OfmapsFoldIdx     = serWaveOp.gOfmapsFoldIdx();
    
                waveOp = new wave::SbAtomSaveWaveOp(sbatomsaveParams, prevWaveOps);
                assert(waveOp->gName() == sbatomsaveParams.m_WaveOpName);
    
            } else if (serWaveOp.gWaveOpType() == wave::MatMulWaveOp::gTypeStr()) {
                wave::MatMulWaveOp::Params matmulParams;
                fillWaveOpParams(serWaveOp, matmulParams);
    
                matmulParams.m_BatchingInWave       = serWaveOp.gBatchingInWave();
                matmulParams.m_IfmapCount           = serWaveOp.gIfmapCount();
                matmulParams.m_IfmapTileHeight      = serWaveOp.gIfmapTileHeight();
                matmulParams.m_IfmapTileWidth       = serWaveOp.gIfmapTileWidth();
                matmulParams.m_IfmapsAtomId         = serWaveOp.gIfmapsAtomId();
                matmulParams.m_IfmapsOffsetInAtom   = serWaveOp.gIfmapsOffsetInAtom();
                // layer_name
                matmulParams.m_OfmapCount           = serWaveOp.gOfmapCount();
                matmulParams.m_OfmapTileHeight      = serWaveOp.gOfmapTileHeight();
                matmulParams.m_OfmapTileWidth       = serWaveOp.gOfmapTileWidth();
                // previous layers
                matmulParams.m_PsumBankId           = serWaveOp.gPsumBankId();
                matmulParams.m_PsumBankOffset       = serWaveOp.gPsumBankOffset();
                matmulParams.m_Start                = serWaveOp.qStart();
                matmulParams.m_WaveId               = serWaveOp.gWaveId();
                matmulParams.m_WaveIdFormat         = serWaveOp.gWaveIdFormat();
                // waveop name
                // waveop type
                matmulParams.m_WeightsAtomId        = serWaveOp.gWeightsAtomId();
                matmulParams.m_WeightsOffsetInAtom  = serWaveOp.gWeightsOffsetInAtom();
    
                waveOp = new wave::MatMulWaveOp(matmulParams, prevWaveOps);
                assert(waveOp->gName() == matmulParams.m_WaveOpName);
    
            } else {
                assert(false && "Wrong WaveOp type during deserialization");
            }
    
            m_WaveOps.push_back(waveOp);
            assert(m_Name2WaveOp.find(waveOp->gName()) == m_Name2WaveOp.end());
            m_Name2WaveOp[waveOp->gName()] = waveOp;
        }
    }
}

}}



