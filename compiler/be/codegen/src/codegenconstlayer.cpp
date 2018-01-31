#include "constlayer.hpp"

#include "codegenconstlayer.hpp"

//-----------------------------------------------------------------
// *[ifmap/filter]Addrs are arrays of statebuffer addresses.  Arrays
// * deal with cases iwhen with the number of ifmap channels is > number of rows.
// * In this case, the ifmaps and filters must be "wrapped".  Each address in the
// * array is the wrap offset
//-----------------------------------------------------------------

namespace kcc {
namespace codegen {

void
CodeGenConstLayer::generate(layers::Layer* layer)
{
    const auto constLayer = dynamic_cast<layers::ConstLayer*>(layer);
    assert(constLayer && "CodeGenLayer::generate: layer is not a Const layer");
    Generate(constLayer, constLayer->gOfmapAddress());
}

}}




