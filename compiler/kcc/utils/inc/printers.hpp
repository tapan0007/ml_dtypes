#pragma once

#ifndef KCC_UTILS_PRINTERS_H
#define KCC_UTILS_PRINTERS_H 1

#include "consts.hpp"
#include "datatype.hpp"
#include "layer.hpp"
#include "network.hpp"

namespace kcc {
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

    void printJsonOld(obj, filename):

    void printJson(self, obj, filename):
};

}}

#endif // KCC_UTILS_PRINTERS_H

