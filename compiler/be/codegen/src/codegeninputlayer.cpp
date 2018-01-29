#include "inputlayer.hpp"

#include "codegeninputlayer.hpp"

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


