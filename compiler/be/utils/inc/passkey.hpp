#pragma once

#ifndef KCC_UTILS_PASSKEY_H
#define KCC_UTILS_PASSKEY_H

namespace kcc {
namespace utils {

template <typename T>
class Passkey {
private:
    friend T;

    Passkey() {}  // Must not be default
    Passkey(const Passkey&) {} // Must not be default

    Passkey& operator=(const Passkey&) = delete;
};


}} // namespace utils, kcc

#endif

