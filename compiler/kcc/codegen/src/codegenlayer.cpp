#include "network.hpp"

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
CodeGenLayer::epilogue(const Layer* const layer)
{
    if ((layer->gRefFileName() != "") || (layer->gNumNextLayers()==0)) {
        char outNpyFileName[256];

        sprintf(outNpyFileName, "%s-%s-simout.npy",
                 layer->gNetwork()->gName().c_str(),
                 layer->gName().c_str());

        for (char* p = outNpyFileName; *p; ++p) {
            if ('/' == *p) {
                *p = '-';
            } else {
                *p = tolower(*p);
            }
        }

        FILE* const objFile = gObjFile();
        const ARBPRECTYPE outDataType = layer->gDataType().gTypeId();
        compile_write_ofmap(objFile,
                outNpyFileName,
                m_OfmapAddrs, m_OfmapDims, outDataType);
    }
}

}}

