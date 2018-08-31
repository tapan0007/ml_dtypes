#pragma once

#ifndef KCC_UTILS_VERSION_H
#define KCC_UTILS_VERSION_H

namespace kcc {
namespace utils {

class Git {
public:
    static const char* gShaLong();
    static const char* gShaShort();
};

}}

#endif

