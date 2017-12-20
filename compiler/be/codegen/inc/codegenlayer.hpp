#pragma once

#ifndef KCC_CODEGEN_CODEGENLAYER_H
#define KCC_CODEGEN_CODEGENLAYER_H

#include <string>
#include <cstdio>


using namespace std;

#include "tcc.hpp"

#include "consts.hpp"
#include "types.hpp"


namespace kcc {

namespace layers {
    class Layer;
}

namespace codegen {
using layers::Layer;

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
    virtual void generate(Layer* layer) = 0;

    //----------------------------------------------------------------
    Layer* gLayer() const;

    //----------------------------------------------------------------
    void rLayer(Layer* layer) {
        m_Layer = layer;
    }

protected:
    void epilogue(const Layer* layer);

protected:
    // These are variables because format can change
    kcc_int32 m_FmapIndex_N = FmapIndex_N; // batch",
    kcc_int32 m_FmapIndex_C = FmapIndex_C; // num ifmaps",
    kcc_int32 m_FmapIndex_H = FmapIndex_H; // ifmap height",
    kcc_int32 m_FmapIndex_W = FmapIndex_W; // ifmap width",

    kcc_int32 m_FilterIndex_M = FilterIndex_M; // filter num ofmaps",
    kcc_int32 m_FilterIndex_C = FilterIndex_C; // filter num ifmaps",
    kcc_int32 m_FilterIndex_R = FilterIndex_R; // filter height",
    kcc_int32 m_FilterIndex_S = FilterIndex_S; // filter width",

    CodeGen* const m_CodeGen;
    Layer* m_Layer;

    addr_t   m_OfmapAddrs;
    uint64_t m_OfmapDims[FMAP_TENSOR_RANK];
    string   m_OfmapFormat;

    addr_t   m_IfmapAddrs[2] = {0, 0};// 2 is tmp for single Ifmap
    uint64_t m_IfmapDims[FMAP_TENSOR_RANK];
    string   m_IfmapFormat;
    string   m_IfmapFileName;

    addr_t   m_FilterAddr[2];
    uint64_t m_FilterDims[FMAP_TENSOR_RANK];
    string   m_FilterFormat;
    string   m_FilterFileNames[2]; // more when #ofmaps > #cols

    uint8_t  m_Padding[4]; // From TCC
    uint8_t  m_Dilate[2]; // From TCC
};

}}

#endif // KCC_CODEGEN_CODEGENLAYER_H

