#include <limits>
#include <sstream>

#include "utils/inc/asserter.hpp"
#include "utils/inc/debug.hpp"

#include "dma/inc/dmaqueue.hpp"


namespace kcc {
namespace dma {


/***********************************************************************
***********************************************************************/
DmaQueue::DmaQueue(const std::string& queName, EngineId engId,
                   QueueType typ, kcc_int32 semId)
    : m_Name(queName)
    , m_EngineId(engId)
    , m_QueueType(typ)
    , m_SemaphoreId(semId)
    , m_Count(0)
{
    Assert(typ != QueueType::None, "Bad queue type");
}


}}

