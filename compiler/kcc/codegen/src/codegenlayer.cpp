#include "codegen.hpp"
#include "codegenlayer.hpp"

namespace kcc {
namespace codegen {

//----------------------------------------------------------------
FILE*
CodeGenLayer::gObjFile() const
{
    return m_CodeGen->gObjFile();
}

//----------------------------------------------------------------
Layer*
CodeGenLayer::gLayer() const
{
    return m_Layer;
}

}}

