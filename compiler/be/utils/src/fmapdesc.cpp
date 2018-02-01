
#include <sstream>

#include "fmapdesc.hpp"

namespace kcc {
namespace utils {

//----------------------------------------------------------------
std::string
FmapDesc::gString() const
{
    std::stringstream ss;
    ss  << "(" << gNumMaps() << "," << gMapHeight() << '*' << gMapWidth() << ")";
    return ss.str();
}

}}

