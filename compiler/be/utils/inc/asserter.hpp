#pragma once

#ifndef KCC_UTILS_ASSERTER_H

namespace kcc {
namespace utils {

// Usage
// #define KCC_ASSERT(
//
// KCC_ASSERT(a==b, "Value a=%d  !=  b=%d", a, b);
// Asserter(__LINE__, __FILE__)(a==b, "a==b", "Value a=%d  !=  b=%d", a, b);

class Asserter {
public:
    Asserter(int lineNum, const char* fileName);

    void operator() (bool expr, const char* exprStr, const char* fmt, ...);

private:
    void crash(const char*);

    Asserter() = delete;
    Asserter(const Asserter&) = delete;
    Asserter& operator= (const Asserter&) = delete;

private:
    const int           m_LineNumber;
    const char* const   m_FileName;
};

#define Assert(expr, fmt, ...) kcc::utils::Asserter(__LINE__, __FILE__)(expr, #expr, fmt, __VA_ARGS__)


}}


#endif

