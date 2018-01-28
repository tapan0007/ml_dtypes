#include "avgpoollayer.hpp"

#include "codegenavgpoollayer.hpp"

//-----------------------------------------------------------------
// *[ifmap/filter]Addrs are arrays of statebuffer addresses.  Arrays
// * deal with cases iwhen with the number of ifmap channels is > number of rows.
// * In this case, the ifmaps and filters must be "wrapped".  Each address in the
// * array is the wrap offset
//-----------------------------------------------------------------

namespace kcc {
namespace codegen {

void
CodeGenAvgPoolLayer::generate(layers::Layer* layer)
{
    layers::AvgPoolLayer* const avgpoolLayer = dynamic_cast<layers::AvgPoolLayer*>(layer);
    assert(avgpoolLayer && "CodeGenLayer::generate: layer is not an AvgPool layer");
    Generate(layer, POOLFUNC::AVG_POOL);
}

}}




