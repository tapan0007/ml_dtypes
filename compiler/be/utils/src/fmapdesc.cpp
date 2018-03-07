
#include <sstream>

#include "utils/inc/asserter.hpp"
#include "utils/inc/fmapdesc.hpp"

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

