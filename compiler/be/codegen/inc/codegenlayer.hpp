#pragma once

#ifndef KCC_CODEGEN_CODEGENLAYER_H
#define KCC_CODEGEN_CODEGENLAYER_H

#include <string>
#include <cstdio>



#include "tcc/inc/tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace layers {
    class Layer;
}

namespace codegen {

class CodeGen;


class CodeGenLayer {
public:
    //----------------------------------------------------------------
    CodeGenLayer(CodeGen* codegen)
        : m_CodeGen(codegen)
    {}

    virtual ~CodeGenLayer()
    {}

    //----------------------------------------------------------------
    FILE* gObjFile() const;

    //----------------------------------------------------------------
    virtual void generate(layers::Layer* layer) = 0;

    //----------------------------------------------------------------
    layers::Layer* gLayer() const;

    //----------------------------------------------------------------
    void rLayer(layers::Layer* layer) {
        m_Layer = layer;
    }

protected:
    void epilogue(const layers::Layer* layer);

protected:
    // These are variables because format can change
    kcc_int32 m_FmapIndex_N = FmapIndex_N; // batch",
    kcc_int32 m_FmapIndex_C = FmapIndex_C; // num ifmaps",
    kcc_int32 m_FmapIndex_H = FmapIndex_H; // ifmap height",
    kcc_int32 m_FmapIndex_W = FmapIndex_W; // ifmap width",

    CodeGen* const m_CodeGen;
    layers::Layer* m_Layer;

    addr_t   m_OfmapAddrs;
    uint64_t m_OfmapDims[FMAP_TENSOR_RANK];
    std::string   m_OfmapFormat;

    addr_t   m_IfmapAddrs[2] = {0, 0};// 2 is tmp for single Ifmap
    uint64_t m_IfmapDims[FMAP_TENSOR_RANK];
    std::string   m_IfmapFormat;
    std::string   m_IfmapFileName;

    uint8_t  m_Padding[4]; // From TCC
    uint8_t  m_Dilate[2]; // From TCC
};

}}

#endif // KCC_CODEGEN_CODEGENLAYER_H

