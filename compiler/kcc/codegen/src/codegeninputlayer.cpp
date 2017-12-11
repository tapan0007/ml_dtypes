#include "inputlayer.hpp"

#include "codegeninputlayer.hpp"

namespace kcc {
using layers::InputLayer;

namespace codegen {

//----------------------------------------------------------------
void
CodeGenInputLayer::generate(Layer* layer)
{
    InputLayer* inLayer = dynamic_cast<InputLayer*>(layer);
    assert(inLayer);
    FILE* objFile = gObjFile();
    m_Ifmap_addrs[0] = inLayer->gOfmapAddress();
    compile_read_ifmap(objFile,
              m_Ifmap_addrs[0],
              inLayer->gInputDataFileName().c_str(),
              inLayer->gDataTensorDimSemantics().c_str());
}


}}


