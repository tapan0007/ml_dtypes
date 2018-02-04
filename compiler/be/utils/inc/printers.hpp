#pragma once

#ifndef KCC_UTILS_PRINTERS_H
#define KCC_UTILS_PRINTERS_H 1

#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"

namespace kcc {
namespace nest {
    class Network;
}
namespace schedule {
    class Scheduler;
}

namespace utils {

class Printer {
public:
    //--------------------------------------------------------
    Printer(nets::Network* netwk)
        : m_Network(netwk)
    {}

    //--------------------------------------------------------
    void printNetwork();

    //--------------------------------------------------------
    void printDot();

    //--------------------------------------------------------
    void printLevels(schedule::Scheduler* scheduler);

    //--------------------------------------------------------
    void printSched();


private:
    nets::Network* m_Network;
    layers::Layer*   m_PrevLayer;
};

}}

#endif // KCC_UTILS_PRINTERS_H

