#pragma once

#ifndef KCC_UTILS_PRINTERS_H
#define KCC_UTILS_PRINTERS_H 1

#include "consts.hpp"
#include "datatype.hpp"
#include "layer.hpp"
//#include "network.hpp"

namespace kcc {
namespace nest {
    class Network;
}
namespace schedule {
    class Scheduler;
}
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
    void printLevels(schedule::Scheduler* scheduler);

    //--------------------------------------------------------
    void printSched();


private:
    Network* m_Network;
    Layer*   m_PrevLayer;
};

}}

#endif // KCC_UTILS_PRINTERS_H

