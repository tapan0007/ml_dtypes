#include <cstdarg>
#include <cstdio>
#include <assert.h>

#include "utils/inc/asserter.hpp"

namespace kcc {
namespace utils {



Asserter::Asserter(int lineNum, const char* fileName)
    : m_LineNumber(lineNum)
    , m_FileName(fileName)
{
}

void
Asserter::operator() (bool expr, const char* exprStr, const char* fmt, ...)
{
    if (! expr) {
        fprintf(stderr, "File %s:%d assert(%s): ", m_FileName, m_LineNumber, exprStr);

        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);

        assert(false);
    }
}


}}


