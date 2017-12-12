#pragma once

#ifndef KCC_UTILS_PRINTERS_H
#define KCC_UTILS_PRINTERS_H 1

#include "consts.hpp"
#include "datatype.hpp"
#include "layer.hpp"
#include "network.hpp"

namespace kcc {
using nets::Network;
using layers::Layer;

namespace utils {

class Printer {
public:
    //--------------------------------------------------------
    Printer(Network* netwk)
        : m_Network(netwk)
    {}

    //--------------------------------------------------------
    void printNetwork();

    //--------------------------------------------------------
    void printDot();

    //--------------------------------------------------------
    void printLevels();

    //--------------------------------------------------------
    void printSched();

#if 0
    void printJsonOld(obj, filename):

    void printJson(self, obj, filename):
#endif

private:
    Network* m_Network;
    Layer*   m_PrevLayer;
};

}}

#endif // KCC_UTILS_PRINTERS_H

