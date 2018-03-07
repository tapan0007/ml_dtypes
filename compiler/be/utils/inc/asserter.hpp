#pragma once

#ifndef KCC_UTILS_ASSERTER_H
#define KCC_UTILS_ASSERTER_H 1

#include <iostream>

namespace kcc {
namespace utils {



//template<typename T, typename... Targs>
class Asserter {
private:
    enum { BUF_SIZE = 1024 };

public:
    Asserter (int lineNum, const char* fileName, const char* exprStr);

    void operator() (bool expr) const;

    template<typename T, typename... Targs>
    inline void operator() (bool expr, T arg, Targs... Fargs) // recursive variadic function
    {
        if (expr) {
            return;
        }
        char buf[BUF_SIZE];
        snprintf(buf, sizeof(buf)/sizeof(buf[0]),
            "error: File %s:%d, Assertion '%s' failed: ", m_FileName, m_LineNumber, m_ExprStr);
        this->printer(buf, arg, Fargs...);
        crash();
    }

private:
    void crash() const;
    void printer () const;

    template<typename T, typename... Targs>
    inline void printer (T arg, Targs... Fargs) const
    {
        std::cerr << arg;
        this->printer(Fargs...);
    }

private:
    Asserter () = delete;
    Asserter (const Asserter&) = delete;
    Asserter& operator= (const Asserter&) = delete;

private:
    const int           m_LineNumber;
    const char* const   m_FileName;
    const char* const   m_ExprStr;
}; // class Asserter

}}

#define Assert1(expr) kcc::utils::Asserter(__LINE__,__FILE__,#expr)(expr)
#define Assert(expr,...) kcc::utils::Asserter(__LINE__,__FILE__,#expr)(expr,__VA_ARGS__)

// __VA_OPT__ in macros does not work, so asserting without comments (discouraged) needs
// a separate macro

#endif


