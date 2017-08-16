#include "types.h"

const unsigned int Constants::columns = 128;
const unsigned int Constants::rows = 64;
const unsigned int Constants::banks_per_partition = 4;
const size_t Constants::bytes_per_bank = 8192;
const size_t Constants::partition_nbytes = Constants::banks_per_partition * Constants::bytes_per_bank;
