
#include "codegen/inc/codegenrelulayer.hpp"

namespace kcc {
namespace codegen {

ACTIVATIONFUNC
CodeGenReluLayer::gActivFunc() const
{
    return ACTIVATIONFUNC::RELU;
}

}}
