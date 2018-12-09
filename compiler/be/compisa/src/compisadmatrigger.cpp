
//#include "runtime/isa/common/aws_tonga_isa_common.h"


#include "utils/inc/asserter.hpp"

#include "events/inc/events.hpp"

#include "compisa/inc/compisadmatrigger.hpp"

namespace kcc {
namespace compisa {


//****************************************************************
void
DmaTriggerInstr::SetDmaQueueName(const char* queName)
{
    static_assert(sizeof(DmaTriggerInstr) == sizeof(DmaTriggerInstrBase::BaseClass),
        "DmaTriggerInstr class size not equal to base class size");

    enum : kcc_uint64 {
        dmaQueueMaxNameSize = sizeof(dma_queue_name)/sizeof(dma_queue_name[0])
    };
    Assert(std::strlen(queName) < dmaQueueMaxNameSize,
           "Queue name too long: ", queName);
    std::strncpy(dma_queue_name, queName, dmaQueueMaxNameSize - 1);
}


}}

