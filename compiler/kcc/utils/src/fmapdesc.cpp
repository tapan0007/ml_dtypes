
#include <sstream>

#include "fmapdesc.h"

namespace kcc {
namespace utils {

//----------------------------------------------------------------
string
FmapDesc::gString() const
{
    std::stringstream ss;
    ss  << "(" << gNumMaps() << "," << gMapHeight() << '*' << gMapWidth() << ")";
    return ss.str();
}

}}

