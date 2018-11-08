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
    std::cerr << "ERROR: File " << m_FileName << ":" << m_LineNumber << ", Assertion '" << m_ExprStr << "' failed\n";
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


