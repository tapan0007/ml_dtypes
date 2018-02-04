
#include "layers/inc/resaddlayer.hpp"
#include "codegen/inc/codegenresaddlayer.hpp"

namespace kcc {
namespace codegen {

void
CodeGenResAddLayer::generate(layers::Layer* layer)
{
    FILE* const objFile = gObjFile();
    assert(objFile);
    layers::ResAddLayer* const resaddLayer = dynamic_cast<layers::ResAddLayer*>(layer);
    assert(resaddLayer && "CodeGenLayer::generate: layer is not a ResAdd layer");
}

}}

