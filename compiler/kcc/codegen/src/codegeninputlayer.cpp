#include "inputlayer.hpp"

#include "codegeninputlayer.hpp"

namespace kcc {
using layers::InputLayer;

namespace codegen {

//----------------------------------------------------------------
void
CodeGenInputLayer::generate(Layer* layer)
{
    FILE* const objFile = gObjFile();
    InputLayer* const inLayer = dynamic_cast<InputLayer*>(layer);
    assert(inLayer);
    m_IfmapAddrs[0] = inLayer->gOfmapAddress();
    compile_read_ifmap(objFile,
              m_IfmapAddrs[0],
              inLayer->gInputDataFileName().c_str(),
              inLayer->gDataTensorDimSemantics().c_str());
}


}}


