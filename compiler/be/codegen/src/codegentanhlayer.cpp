
#include "codegen/inc/codegentanhlayer.hpp"

namespace kcc {
namespace codegen {

ACTIVATIONFUNC
CodeGenTanhLayer::gActivFunc() const
{
    return ACTIVATIONFUNC::TANH;
}

}}

