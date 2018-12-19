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
                   QueueType typ, kcc_int32 semId, bool firstQue)
    : m_Name(queName)
    , m_EngineId(engId)
    , m_QueueType(typ)
    , m_SemaphoreId(semId)
    , m_Count(0)
    , m_FirstQueue(firstQue)
{
    Assert(typ != QueueType::None, "Bad queue type");
}


}}

