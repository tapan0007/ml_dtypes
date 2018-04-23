
#include "codegen/inc/codegenrelulayer.hpp"

namespace kcc {
namespace codegen {

TONGA_ISA_TPB_ACTIVATION_FUNC
CodeGenReluLayer::gActivFunc() const
{
    return TONGA_ISA_TPB_ACTIVATION_FUNC_RELU;
}

}}
