#include "inputlayer.hpp"

#include "codegendatalayer.hpp"

namespace kcc {

namespace codegen {

//----------------------------------------------------------------
void
CodeGenDataLayer::Generate(layers::DataLayer* dataLayer, addr_t sbAddress)
{
    FILE* const objFile = gObjFile();
    compile_read_ifmap(objFile,
              sbAddress,
              dataLayer->gRefFileName().c_str(),
              dataLayer->gRefFileFormat().c_str());
}


}}



