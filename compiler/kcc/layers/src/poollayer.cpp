
#include "poollayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
std::string
PoolLayer::gPoolLayerStr() const
{
    std::stringstream ss;
    ss << gKernelHeight() << "*" <<  gKernelWidth();

    ss << gName()  // << "{" << gTypeStr() << "}"
       << gBaseLayerStr() 
       << ", kernel=" << gKernelHeight() << "*" << gKernelWidth()
       << ", stride=" << gStrideBT() << "*" << gStrideLR()
       << gStateSizesStr();
    return ss.str();
}


}}

