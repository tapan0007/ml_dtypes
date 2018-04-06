#pragma once

#ifndef KCC_COMPISA_COMMON_H
#define KCC_COMPISA_COMMON_H

//#include "tpb_isa.hpp"
//#include "events/inc/events.hpp"

struct TPB_CMD_SYNC;

namespace kcc {
namespace compisa {

void InitSync(TPB_CMD_SYNC& sync);

}}

#endif // KCC_COMPISA_COMMON_H

