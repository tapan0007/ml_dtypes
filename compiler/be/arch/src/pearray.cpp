
#include "arch/inc/pearray.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
PeArray::PeArray(kcc_int32 numberRows, kcc_int32 numberColumns)
    : m_NumberRows(numberRows)
    , m_NumberColumns(numberColumns)
{
    assert(numberRows > 0 && numberColumns > 0 && "Number of rows or columns not positive");
    if (numberRows > numberColumns) {
        assert(numberRows % numberColumns == 0 && "Number of rows > numer of columns, but not multiple of it");
    } else if (numberRows < numberColumns) {
        assert(numberColumns % numberRows == 0 && "Number of columns > number of rows, but not multiple of it");
    }

}

}}

