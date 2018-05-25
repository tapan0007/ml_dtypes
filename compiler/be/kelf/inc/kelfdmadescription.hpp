#pragma once

#include <string>
#include <vector>
#include <map>


#include "utils/inc/types.hpp"
#include "events/inc/events.hpp"

namespace kcc {
namespace kelf {



/***********************************************************************
***********************************************************************/
class DmaDescription {
public:
    class DmaBlockToTpb;
    class DmaBlockFromTpb;
    class DmaBlockInput;

private:
    class DmaBlock;
    class DmaBlockNonIo;
    class DmaDescToTpb;
    class DmaDescFromTpb;
    using FileIdType = std::string;


public:
    DmaDescription();

public:
    DmaBlockToTpb&      startNewDmaBlockToTpb(EngineId engId, bool qWeights);
    DmaBlockFromTpb&    startNewDmaBlockFromTpb(EngineId engId, bool qOut);
    DmaBlockInput&      startNewDmaBlockInput();

    void writeDmaDescriptors(const char* binFileName, EngineId engId);

    void writeInOutDescriptors();
    void writeDefinitions();

private:
    static const char* gSymbolicOutput();

    static const char* gSymbolicInQueue();
    static const char* gSymbolicOutQueue();

private:
    FileIdType gFileSymbolicId(const std::string& fileName);
    static const char* gSymbolicStateBuffer();
    static const char* gSymbolicInput();

    std::string gSymbolicQueue(EngineId engId, bool inp, bool weight) const;

    const char* gJsonFileName(EngineId engId);
    const char* gEngineName(EngineId engId);

    kcc_int32 gBlockIdForQueue(const std::string&);
    kcc_int32 gNumBlockIdsForQueue(const std::string&) const;

private:
    kcc_int32                           m_FileIdCnt = 0;
    std::map<std::string, FileIdType>   m_FileNameToId;
    std::map<std::string, kcc_int32>    m_QueueToBlockId;
    std::vector<DmaBlockToTpb>          m_DmaBlocksToTpb;
    std::vector<DmaBlockFromTpb>        m_DmaBlocksFromTpb;
    std::vector<DmaBlockInput>          m_DmaBlocksInput;

    const char* const                   m_PeJsonFileName    = "pe.json";
    const char* const                   m_ActJsonFileName   = "act.json";
    const char* const                   m_PoolJsonFileName  = "pool.json";
    const char* const                   m_HostJsonFileName  = "host.json";
    const char* const                   m_DefJsonFileName   = "def.json";
}; // class DmaDescription


/***********************************************************************
***********************************************************************/
class DmaDescription::DmaDescToTpb {
public:
    TpbAddress gDstSbAddress() const {
        return m_DstSbAddress;
    }
    TongaAddress g_SrcFileAddress() const {
        return m_SrcFileAddress;
    }
    kcc_int32 gNumBytes() const {
        return m_NumBytes;
    }
public:
    kcc_int32     m_NumBytes;
    TpbAddress    m_DstSbAddress;
    TongaAddress  m_SrcFileAddress;
    FileIdType    m_SrcFileId;
};


/***********************************************************************
***********************************************************************/
class DmaDescription::DmaDescFromTpb {
public:
    TpbAddress gDstFileAddress() const {
        return m_DstFileAddress;
    }
    TpbAddress gSrcSbAddress() const {
        return m_SrcSbAddress;
    }
public:
    kcc_int32       m_NumBytes;
    TpbAddress      m_SrcSbAddress;
    TongaAddress    m_DstFileAddress;
    FileIdType      m_DstFileId;
};



/***********************************************************************
***********************************************************************/
class DmaDescription:: DmaBlock {
public:
    DmaBlock(DmaDescription& dmaDescription);
    DmaBlock() = delete;

    void addTailEventId(events::EventId eventId);

    kcc_int32 gBlockId() const {
        return m_BlockId;
    }
    events::EventId gEventId() const {
        return m_TailEventIds[0];
    }
    const std::string& gQueueName() const {
        return m_QueueName;
    }

protected:
    DmaDescription&                 m_DmaDescription;
    std::vector<events::EventId>    m_TailEventIds;
    kcc_int32                       m_BlockId;
    std::string                     m_QueueName;
}; // class DmaDescription::DmaBlock


/***********************************************************************
***********************************************************************/
class DmaDescription::DmaBlockNonIo : public DmaBlock {
public:
    DmaBlockNonIo(DmaDescription& dmaDescription, EngineId engId);
    DmaBlockNonIo() = delete;

    EngineId gTriggerEngineId() const {
        return m_EngineId;
    }

protected:
    const EngineId          m_EngineId;
}; // class DmaDescription::DmaBlockNonIo


/***********************************************************************
***********************************************************************/
class DmaDescription::DmaBlockInput : public DmaBlock {
public:
    DmaBlockInput(DmaDescription& dmaDescription);
    DmaBlockInput() = delete;

    void addDmaDesc(kcc_uint64 srcFileAddress,
                TongaAddress dstTongaAddress, kcc_int32 numBytes);
    kcc_int32 gId() const;

    //std::vector<DmaDescToTpb> 
    const auto& gDescs() const {
        return m_Descs;
    }

private:
    std::vector<DmaDescToTpb> m_Descs;
}; // class DmaDescription::DmaBlockInput



/***********************************************************************
***********************************************************************/
class DmaDescription::DmaBlockToTpb : public DmaDescription::DmaBlockNonIo {
public:
    DmaBlockToTpb(DmaDescription& dmaDescription, EngineId engId, bool qWeights);
    DmaBlockToTpb() = delete;

    void addDmaDesc(TongaAddress srcFileAddress, const std::string& refFile,
                    TpbAddress dstSbAddress, kcc_int32 numBytes);
    kcc_int32 gId() const;



    bool qWeights() const {
        return m_QWeights;
    }
    const auto& gDescs() const {
        return m_Descs;
    }
    std::string gSymbolicQueueName(EngineId engId) const {
        return m_DmaDescription.gSymbolicQueue(engId, true, m_QWeights);
    }

private:
    bool m_QWeights;
    std::vector<DmaDescToTpb> m_Descs;
}; // class DmaDescription::DmaBlockToTpb



/***********************************************************************
***********************************************************************/
class DmaDescription::DmaBlockFromTpb : public DmaDescription::DmaBlockNonIo {
public:
    DmaBlockFromTpb(DmaDescription& dmaDescription, EngineId engId, bool qOut);
    DmaBlockFromTpb() = delete;

    kcc_int32 gId() const;

    void addDmaDesc(TpbAddress srcSbAddress, TongaAddress dstFileAddress,
                    const std::string& refFile, kcc_int32 numBytes);

    const auto& gDescs() const {
        return m_Descs;
    }

    std::string gSymbolicQueueName(EngineId engId) const {
        return m_DmaDescription.gSymbolicQueue(engId, false, false);
    }

    bool qOut() const {
        return m_QOut;
    }

private:
    bool    m_QOut;
    std::vector<DmaDescFromTpb> m_Descs;
}; // class DmaDescription::DmaBlockFromTpb


}}

