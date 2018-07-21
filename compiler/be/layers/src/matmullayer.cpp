#include <sstream>

#include "layers/inc/matmullayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
MatmulLayer::MatmulLayer(const MatmulLayer::Params& params, Layer* prev_layer,
        const FmapDesc& fmapDesc,
        const std::tuple<kcc_int32,kcc_int32>& kernel,
        const char* filterFileName, const char* filterTensorFormat)
    : Layer(params, fmapDesc, std::vector<Layer*>(1, prev_layer))
{
    m_Kernel.m_Height = std::get<0>(kernel);
    m_Kernel.m_Width  = std::get<1>(kernel);
    m_FilterFileName           = filterFileName;
    m_FilterTensorFormat       = filterTensorFormat;
}


//----------------------------------------------------------------
const char*
MatmulLayer::gTypeStr() const
{
    return TypeStr();
}

std::string
MatmulLayer::gString() const
{
    return TypeStr();
}

bool
MatmulLayer::verify() const
{
    if (! this->Layer::verify()) {
        return false;
    }
    return true;
}


MatmulLayer::Params::Params(const Layer::Params& params)
    : Layer::Params(params)
{
    m_LayerName     = params.m_LayerName;
    m_BatchFactor   = params.m_BatchFactor;
    m_Network       = params.m_Network;
    m_RefFile       = params.m_RefFile;
    m_RefFileFormat = params.m_RefFileFormat;
}


}}


