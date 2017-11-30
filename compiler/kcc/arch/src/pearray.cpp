
#include "pearray.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
PeArray::PeArray(int numberRows, int numberColumns)
{
    assert(numberRows > 0 and numberColumns > 0);
    if (numberRows > numberColumns) {
        assert(numberRows % numberColumns == 0);
    } else if (numberRows < numberColumns) {
        assert(numberColumns % numberRows == 0);
    }

    m_NumberRows    = numberRows;
    m_NumberColumns = numberColumns;
}

}}

