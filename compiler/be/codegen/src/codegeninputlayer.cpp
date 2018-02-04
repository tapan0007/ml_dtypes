#include "layers/inc/inputlayer.hpp"

#include "codegen/inc/codegeninputlayer.hpp"

namespace kcc {

namespace codegen {

//----------------------------------------------------------------
void
CodeGenInputLayer::generate(layers::Layer* layer)
{
    const auto inLayer = dynamic_cast<layers::InputLayer*>(layer);
    assert(inLayer && "CodeGenLayer::generate: layer is not an Input layer");
    Generate(inLayer, inLayer->gOfmapAddress());
}


}}


