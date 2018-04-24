
#include "codegen/inc/codegentanhlayer.hpp"

namespace kcc {
namespace codegen {

TONGA_ISA_TPB_ACTIVATION_FUNC
CodeGenTanhLayer::gActivFunc() const
{
    return TONGA_ISA_TPB_ACTIVATION_FUNC_TANH;
}

}}

