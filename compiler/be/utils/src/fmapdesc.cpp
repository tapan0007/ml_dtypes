
#include <sstream>

#include "fmapdesc.hpp"

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

