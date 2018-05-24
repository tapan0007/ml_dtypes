#include "tcc/inc/tcc.hpp"

#include "layers/inc/inputlayer.hpp"

#include "codegen/inc/codegendatalayer.hpp"

namespace kcc {

namespace codegen {

//----------------------------------------------------------------
void
CodeGenDataLayer::Generate(layers::DataLayer* dataLayer, tonga_addr sbAddress)
{
    FILE* const objFile = gObjFile();
    compile_read_ifmap(objFile,
              sbAddress,
              dataLayer->gRefFileName().c_str(),
              dataLayer->gRefFileFormat().c_str());
}


}}



