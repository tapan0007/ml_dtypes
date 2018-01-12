
#include "pearray.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
PeArray::PeArray(kcc_int32 numberRows, kcc_int32 numberColumns)
    : m_NumberRows(numberRows)
    , m_NumberColumns(numberColumns)
{
    assert(numberRows > 0 and numberColumns > 0);
    if (numberRows > numberColumns) {
        assert(numberRows % numberColumns == 0);
    } else if (numberRows < numberColumns) {
        assert(numberColumns % numberRows == 0);
    }

}

}}

