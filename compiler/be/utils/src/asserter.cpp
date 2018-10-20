#include <cstdarg>
#include <cstdio>
#include <assert.h>

#include "utils/inc/asserter.hpp"



namespace kcc {
namespace utils {


Asserter::Asserter (int lineNum, const char* fileName, const char* exprStr)
    : m_LineNumber(lineNum)
    , m_FileName(fileName)
    , m_ExprStr(exprStr)
    { }


void
Asserter::operator() (bool expr) const
{
    if (expr) {
        return;
    }
    char buf[BUF_SIZE];
    snprintf(buf, sizeof(buf)/sizeof(buf[0]),
        "ERROR: File %s:%d, Assertion '%s' failed ", m_FileName, m_LineNumber, m_ExprStr);
    this->printer(buf);
    crash();
}

void
Asserter::printer () const
{
    std::cerr << "\n";
}

void
Asserter::crash() const
{
    exit(1);
    assert(false);
}

}}


