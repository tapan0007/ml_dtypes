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
    if (expr) {
        return;
    }
    char buf[1024];
    const int numChars = sprintf(buf, "File %s:%d assert(%s): ", m_FileName, m_LineNumber, exprStr);

    va_list args;
    va_start(args, fmt);
    vsprintf(buf + numChars, fmt, args);
    va_end(args);

    crash(buf);
}

void
Asserter::crash(const char* buf)
{
    fprintf(stderr, "%s\n", buf);
    assert(false);
}

}}


