#pragma once

#ifndef KCC_CODEGEN_CODEGENLAYER_H
#define KCC_CODEGEN_CODEGENLAYER_H

#include <string>
#include <cstdio>

using namespace std;

#include "tcc.hpp"

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
    int m_IfmapIndex_N = 0; // batch",
    int m_IfmapIndex_C = 1; // num ifmaps",
    int m_IfmapIndex_H = 2; // ifmap height",
    int m_IfmapIndex_W = 3; // ifmap width",

    int m_OfmapIndex_N = 0; // batch",
    int m_OfmapIndex_C = 1; // num ifmaps",
    int m_OfmapIndex_H = 2; // ifmap height",
    int m_OfmapIndex_W = 3; // ifmap width",

    int m_FilterIndex_M = 0; // filter num ofmaps",
    int m_FilterIndex_C = 1; // filter num ifmaps",
    int m_FilterIndex_R = 2; // filter height",
    int m_FilterIndex_S = 3; // filter width",

    CodeGen* const m_CodeGen;
    Layer* m_Layer;

    addr_t   m_OfmapAddrs;
    uint64_t m_OfmapDims[4];
    string   m_OfmapFormat;

    addr_t   m_IfmapAddrs[2] = {0, 0};// 2 is tmp for single Ifmap
    uint64_t m_IfmapDims[4];
    string   m_IfmapFormat;
    string   m_IfmapFileName;

    addr_t   m_FilterAddr[2];
    uint64_t m_FilterDims[4];
    string   m_FilterFormat;
    string   m_FilterFileNames[2]; // more when #ofmaps > #cols

    uint8_t  m_Convolve_stride[2];
    uint8_t  m_Padding[2];
    uint8_t  m_Dilate[2];
};

}}

#endif // KCC_CODEGEN_CODEGENLAYER_H

