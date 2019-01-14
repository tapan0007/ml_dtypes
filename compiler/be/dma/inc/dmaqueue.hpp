#pragma once
#ifndef KCC_DMA_DMAQUEUE_H
#define KCC_DMA_DMAQUEUE_H

#include <set>

#include "utils/inc/types.hpp"

namespace kcc {
namespace dma {

class DmaQueue {
public:
    enum class QueueType {
        Input,
        Weights,
        Output,
        SbufToSbuf,
        SbufToTmp,
        TmpToSbuf,
        None
    };
public:
    DmaQueue(const std::string& name, EngineId engId, QueueType typ, kcc_int32 semId, bool firstQue);

    DmaQueue(const DmaQueue&) = default;
    DmaQueue() = delete;

    const std::string& gName() const {
        return m_Name;
    }
    EngineId gEngineId() const {
        return m_EngineId;
    }
    QueueType gQueueType() const {
        return m_QueueType;
    }
    kcc_int32 gSemaphoreId() const {
        return m_SemaphoreId;
    }
    bool qFirstQueue() const {
        return m_FirstQueue;
    }

private:
    const std::string m_Name        = "";
    const EngineId    m_EngineId    = EngineId::None;
    const QueueType   m_QueueType   = QueueType::None;
    const kcc_int32   m_SemaphoreId = -1;
    kcc_int32         m_Count       = 0;
    bool              m_FirstQueue  = true;
};

}}

#endif // KCC_DMA_DMAQUEUE_H

