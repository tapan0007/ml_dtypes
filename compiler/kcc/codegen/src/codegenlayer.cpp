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

void
CodeGenLayer::epilogue(Layer* layer)
{
    if (layer->gRefFileName() != "") {
        FILE* const objFile = gObjFile();
        const ARBPRECTYPE outDataType = layer->gDataType().gTypeId();
        compile_write_ofmap(objFile,
                layer->gRefFileName().c_str(),
                m_OfmapAddrs, m_OfmapDims, outDataType);
    }
}

}}

