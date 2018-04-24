#include "aws_tonga_isa_tpb_pool.h"

#include "layers/inc/maxpoollayer.hpp"

#include "codegen/inc/codegenmaxpoollayer.hpp"

//-----------------------------------------------------------------
// *[ifmap/filter]Addrs are arrays of statebuffer addresses.  Arrays
// * deal with cases iwhen with the number of ifmap channels is > number of rows.
// * In this case, the ifmaps and filters must be "wrapped".  Each address in the
// * array is the wrap offset
//-----------------------------------------------------------------

namespace kcc {
namespace codegen {

void
CodeGenMaxPoolLayer::generate(layers::Layer* layer)
{
    layers::MaxPoolLayer* const maxpoolLayer = dynamic_cast<layers::MaxPoolLayer*>(layer);
    assert(maxpoolLayer && "CodeGen::generate: layer is not a MaxPool layer");
    Generate(maxpoolLayer, TONGA_ISA_TPB_POOL_TYPE_MAXPOOL);
}

}}



