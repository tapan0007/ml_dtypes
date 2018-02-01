#include "serlayer.hpp"

namespace kcc {
namespace serialize {

//----------------------------------------------------------------
const std::string&
SerLayer::gName() const
{
    return m_LayerName;
}

} // namespace serialize
} // namespace kcc
