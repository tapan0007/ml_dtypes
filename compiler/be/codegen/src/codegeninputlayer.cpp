#include "inputlayer.hpp"

#include "codegeninputlayer.hpp"

namespace kcc {

namespace codegen {

//----------------------------------------------------------------
void
CodeGenInputLayer::generate(layers::Layer* layer)
{
    FILE* const objFile = gObjFile();
    const auto inLayer = dynamic_cast<layers::InputLayer*>(layer);
    assert(inLayer && "CodeGenLayer::generate: layer is not an Input layer");
    m_IfmapAddrs[0] = inLayer->gOfmapAddress();
    compile_read_ifmap(objFile,
              m_IfmapAddrs[0],
              inLayer->gInputDataFileName().c_str(),
              inLayer->gDataTensorDimSemantics().c_str());
}


}}


