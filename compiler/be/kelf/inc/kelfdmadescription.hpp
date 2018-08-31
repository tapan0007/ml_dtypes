#pragma once

#include <string>
#include <vector>
#include <map>

#include "nlohmann/json.hpp"

#include "utils/inc/types.hpp"
#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

namespace kcc {

namespace nets {
class Network;
}

namespace kelf {



/***********************************************************************
***********************************************************************/
class DmaDescription {
public:
    class DmaBlockToTpb;
    class DmaBlockFromTpb;
    class DmaBlockInput;

private:
    enum class FileType {
        Weight,             // files containing weights
        Input,              // files containing IFMAPs
        LoadSave,           // tmp for data eviction
        Output,             // files for OFMAPs
        Invalid
    };
    class DmaBlock;
    class DmaBlockNonIo;
    class DmaDesc;
    class DmaDescToTpb;
    class DmaDescFromTpb;

    struct FileIdType {
    public:
        FileIdType()
            : m_VarName("INVALID_VAR")
            , m_FileName("INVALID_FILE")
            , m_FileType(FileType::Invalid)
            , m_Size(-1)
        { }

        FileIdType(std::string varName, std::string fileName, FileType type)
            : m_VarName(varName)
            , m_FileName(fileName)
            , m_FileType(type)
            , m_Size(-1)
        { }

        FileIdType(const FileIdType&);

        std::string m_VarName;
        std::string m_FileName;
        FileType    m_FileType;
        kcc_int64   m_Size;
    };


public:
    DmaDescription(const nets::Network& network);
    DmaDescription() = delete;
    DmaDescription(const DmaDescription&) = delete;

public:
    DmaBlockToTpb&      startNewDmaBlockToTpb(EngineId engId, bool qWeights, const char* comment);
    DmaBlockFromTpb&    startNewDmaBlockFromTpb(EngineId engId, bool qOut, const char* comment);
    DmaBlockInput&      startNewDmaBlockInput(const char* comment);

    void writeDmaDescriptors(const char* binFileName, EngineId engId);

    void writeInOutDescriptors();
    void writeDefinitions();

    kcc_int64 gInputSizeBytes() const {
        return m_InputSizeBytes;
    }
    void rInputSizeBytes(kcc_int64 sz) {
        m_InputSizeBytes = sz;
    }


    void rOutputSizeBytes(kcc_int64 sz, const std::string& refFileName);
    kcc_int64 gOutputSizeBytes(const std::string& refFileName);

    bool qHasFile(const std::string& fileName) const;

private:

    static const char* gSymbolicInQueue();
    static const char* gSymbolicOutQueue();

    struct Keys;

private:
    FileIdType& gFileSymbolicId(const std::string& fileName, FileType fileType);
    FileIdType& gFileSymbolicId(const std::string& fileName);
    const FileIdType& gFileSymbolicId(const std::string& fileName) const;

    static const char* gSymbolicStateBuffer();
    static const char* gSymbolicInput();


    std::string gSymbolicQueue(EngineId engId, bool inp, bool weight) const;

    const char* gJsonFileName(EngineId engId);
    const char* gEngineName(EngineId engId);

    kcc_int32 gBlockIdForQueue(const std::string& queName);
    kcc_int32 gNumBlockIdsForQueue(const std::string& queName) const;
private:
    kcc_int32                           m_WeightFileIdCnt = 0;
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

    const nets::Network&                m_Network;

    kcc_int64                           m_InputSizeBytes         = -1;
    kcc_int64                           m_OutputSizeBytes        = -1;
}; // class DmaDescription


struct DmaDescription::Keys {
    static const char* gBinFileName();
    static const char* gWeightFileName();
    static const char* gJsonName();
    static const char* gQueueName();
    static const char* gFromId();
    static const char* gFromOffset();
    static const char* gToId();
    static const char* gToOffset();
    static const char* gSize();
    static const char* gDescriptor();
    static const char* gBlockId();
    static const char* gDescType();
    static const char* gQueueType();
    static const char* gOwner();
    static const char* gDmaBlock();
    static const char* gDmaQueue();

    static const char* gHashComment();
    static const char* gHashBlockSize();
    static const char* gHashTransferType();
    static const char* gHashFileName();
};


/***********************************************************************
***********************************************************************/
class DmaDescription::DmaDesc {
public:
    DmaDesc(kcc_int64 nBytes)
        : m_NumBytes(nBytes)
    {}
    DmaDesc() = delete;

    kcc_int64 gNumBytes() const {
        return m_NumBytes;
    }
    virtual void assertAccessCheck() const = 0;

private:
    kcc_int64 m_NumBytes = 0;
};

/***********************************************************************
***********************************************************************/
class DmaDescription::DmaDescToTpb : public DmaDesc {
public:
    DmaDescToTpb(kcc_uint64 nBytes, TpbAddress    dstSbAddress,
                 TongaAddress  srcFileAddress, const FileIdType&  srcFileId)
        : DmaDesc(nBytes)
        , m_DstSbAddress(dstSbAddress)
        , m_SrcFileAddress(srcFileAddress)
        , m_SrcFileId(srcFileId)
    {
        Assert(FileType::Weight == srcFileId.m_FileType
            || FileType::Input == srcFileId.m_FileType
            || FileType::LoadSave == srcFileId.m_FileType,
            "Wrong file type of Dma descriptor to TPB");
    }

    DmaDescToTpb() = delete;

    TpbAddress gDstSbAddress() const {
        return m_DstSbAddress;
    }
    TongaAddress gSrcFileAddress() const {
        return m_SrcFileAddress;
    }
    const FileIdType&    gSrcFileId () const {
        return m_SrcFileId;
    }
    void assertAccessCheck() const override;
private:
    TpbAddress    m_DstSbAddress;
    TongaAddress  m_SrcFileAddress;
    FileIdType    m_SrcFileId;
};


/***********************************************************************
***********************************************************************/
class DmaDescription::DmaDescFromTpb : public DmaDesc {
public:
    DmaDescFromTpb(kcc_uint64 nBytes, TpbAddress srcSbAddress,
                   TongaAddress    dstFileAddress, const FileIdType& dstFileId)
        : DmaDesc(nBytes)
        , m_SrcSbAddress(srcSbAddress)
        , m_DstFileAddress(dstFileAddress)
        , m_DstFileId(dstFileId)
    {
        Assert(FileType::Output == dstFileId.m_FileType
            || FileType::LoadSave == dstFileId.m_FileType,
            "Wrong file type of Dma descriptor to TPB");
    }

    DmaDescFromTpb() = delete;
    TpbAddress gDstFileAddress() const {
        return m_DstFileAddress;
    }
    TpbAddress gSrcSbAddress() const {
        return m_SrcSbAddress;
    }
    const FileIdType& gDstFileId() const {
        return m_DstFileId;
    }
    void assertAccessCheck() const override;
private:
    TpbAddress      m_SrcSbAddress;
    TongaAddress    m_DstFileAddress;
    FileIdType      m_DstFileId;
};



/***********************************************************************
***********************************************************************/
class DmaDescription:: DmaBlock {
public:
    DmaBlock(DmaDescription& dmaDescription, const char* comment);
    DmaBlock() = delete;

    void addTailEventId(events::EventId eventId);
    void setDmaEventField(nlohmann::json& jDmaBlock) const;

    kcc_int32 gBlockId() const {
        return m_BlockId;
    }
    const std::string& gQueueName() const {
        return m_QueueName;
    }
    const std::string& gComment() const {
        return m_Comment;
    }


protected:
    DmaDescription&                 m_DmaDescription;
    std::vector<events::EventId>    m_TailEventIds;
    kcc_int32                       m_BlockId;
    std::string                     m_QueueName;
    const std::string               m_Comment;
}; // class DmaDescription::DmaBlock


/***********************************************************************
***********************************************************************/
class DmaDescription::DmaBlockNonIo : public DmaBlock {
public:
    DmaBlockNonIo(DmaDescription& dmaDescription, EngineId engId, const char* comment);
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
    DmaBlockInput(DmaDescription& dmaDescription, const char* comment);
    DmaBlockInput() = delete;

    void addDmaDesc(TongaAddress srcFileAddress,
                TongaAddress dstTongaAddress, kcc_int32 numBytes);
    kcc_int32 gId() const;

    //std::vector<DmaDescToTpb>
    const auto& gDescs() const {
        return m_Descs;
    }

    kcc_uint64 size() const;

    static std::string gSymbolicInputQueueName() {
        return std::string(DmaDescription::gSymbolicInQueue());
    }

private:
    std::vector<DmaDescToTpb> m_Descs;
}; // class DmaDescription::DmaBlockInput



/***********************************************************************
***********************************************************************/
class DmaDescription::DmaBlockToTpb : public DmaDescription::DmaBlockNonIo {
public:
    DmaBlockToTpb(DmaDescription& dmaDescription, EngineId engId, bool qWeights, const char* comment);
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

    kcc_uint64 size() const;

private:
    bool m_QWeights;
    std::vector<DmaDescToTpb> m_Descs;
}; // class DmaDescription::DmaBlockToTpb



/***********************************************************************
***********************************************************************/
class DmaDescription::DmaBlockFromTpb : public DmaDescription::DmaBlockNonIo {
public:
    DmaBlockFromTpb(DmaDescription& dmaDescription, EngineId engId, bool qOut, const char* comment);
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

    kcc_uint64 size() const;

private:
    bool    m_QOut;
    std::vector<DmaDescFromTpb> m_Descs;
}; // class DmaDescription::DmaBlockFromTpb


}}

