#include <cstdio>

using namespace std;

#include "network.hpp"
#include "inputlayer.hpp"
#include "convlayer.hpp"
#include "relulayer.hpp"
#include "tanhlayer.hpp"

#include "codegen.hpp"
#include "codegeninputlayer.hpp"
#include "codegenconvlayer.hpp"
#include "codegenrelulayer.hpp"
#include "codegentanhlayer.hpp"

namespace kcc {
using layers::InputLayer;
using layers::ConvLayer;
using layers::ReluLayer;
using layers::TanhLayer;

namespace codegen {


//  ##########################################################
//  void compile_read_ifmap(FILE *out_binary,
//          const addr_t ifmap_sb_addr, const char *in_numpy_fname,
//          const char *numpy_layout);
//
//  void compile_read_filter(FILE *out_binary,
//          const addr_t filter_sb_addr, const char *in_numpy_fname,
//          const char *numpy_layout);
//
//  void compile_write_ofmap(FILE *out_binary,
//          const char *out_numpy_name, const addr_t ofmap_sb_addr,
//          const uint64_t dims[4],
//          const ARBPRECTYPE dtype);
//
//  /*[ifmap/filter]_addrs are arrays of statebuffer addresses.  Arrays
//   * deal with cases iwhen with the number of ifmap channels is > number of rows.
//   * In this case, the ifmaps and filters must be "wrapped".  Each address in the
//   * array is the wrap offset */
//  void
//  compile_convolve(FILE *out_binary,
//          const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], /* NCHW */
//          const addr_t *filter_addr, const uint64_t filter_dims[4], /* MCRS */
//          const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
//          const ARBPRECTYPE in_dtype, const ARBPRECTYPE out_dtype,
//          const uint8_t padding[2],  /* Height,Width */
//          const uint8_t stride[2],   /* Height,Width */
//          const uint8_t dilate[2]);  /* Height,Width */
//
//  void
//  compile_pool(FILE *out_binary,
//          const addr_t ifmap_addr, const uint64_t ifmap_dims[4], /* NCHW */
//          const uint64_t kernel_dims[4], /* NCHW */
//          const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
//          const uint64_t stride_dims[4], /* NCHW */
//          const ARBPRECTYPE dtype,
//          POOLFUNC pool_func);
//  ##########################################################


//########################################################
CodeGen::CodeGen(Network* ntwk, Arch* arch)
{
    m_Network = ntwk;
    m_Arch = arch;
    createGenMap();
}

//----------------------------------------------------------------
void
CodeGen::createGenMap()
{
    m_InputLayer.reset(new CodeGenInputLayer(this));
    m_ConvLayer.reset(new CodeGenConvLayer(this));
    m_ReluLayer.reset(new CodeGenReluLayer(this));
    m_TanhLayer.reset(new CodeGenTanhLayer(this));
    //m_MaxPoolLayer.reset(new CodeGenMaxPoolLayer());
    //m_AvgPoolLayer.reset(new CodeGenAvgPoolLayer());
}

CodeGenLayer*
CodeGen::gGenFunc(const Layer* layer)
{
    if (dynamic_cast<const InputLayer*>(layer)) {
        return m_InputLayer.get();
    } else if (dynamic_cast<const ConvLayer*>(layer)) {
        return m_ConvLayer.get();
    } else if (dynamic_cast<const ReluLayer*>(layer)) {
        return m_ReluLayer.get();
    } else if (dynamic_cast<const TanhLayer*>(layer)) {
        return m_TanhLayer.get();
    } else {
        assert(false);
    }
}

void
CodeGen::generate(const char* objFileName)
{
    m_ObjFile = std::fopen(objFileName, "w");
    assert(m_ObjFile);

    for (auto layer : m_Network->gSchedForwLayers()) {
        CodeGenLayer* layerGen = gGenFunc(layer);
        layerGen->generate(layer);
    }

    fclose(m_ObjFile); m_ObjFile = nullptr;
}



}}



