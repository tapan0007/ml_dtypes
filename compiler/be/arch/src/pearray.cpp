#include "utils/inc/asserter.hpp"

#include "arch/inc/pearray.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
PeArray::PeArray(kcc_int32 numberRows, kcc_int32 numberColumns, const Arch& arch)
    : ArchEng(arch)
    , m_NumberRows(numberRows)
    , m_NumberColumns(numberColumns)
{
    Assert(numberRows > 0 && numberColumns > 0,  "Number of rows or columns not positive: number_rows=", numberRows, ", number_columns=", numberColumns );
    if (numberRows > numberColumns) {
        Assert(numberRows % numberColumns == 0, "Number of rows > numer of columns, but not multiple of it: number_rows=", numberRows, ", number_columns=", numberColumns);
    } else if (numberRows < numberColumns) {
        Assert(numberColumns % numberRows == 0, "Number of columns > number of rows, but not multiple of it: number_rows=", numberRows, ", number_columns=", numberColumns);
    }

}

}}

