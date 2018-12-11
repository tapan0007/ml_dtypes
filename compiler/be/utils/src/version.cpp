
#include "utils/inc/version.hpp"

namespace kcc {
namespace utils {


const char*
Git::gShaShort()
{
    static const char gitShaShort[] = GIT_SHA_SHORT;
    return gitShaShort;
}

const char*
Git::gShaLong()
{
    static const char gitShaLong[] = GIT_SHA_LONG;
    return gitShaLong;
}

}}

