#include <fstream>
#include <iostream>


// Json package
#include "nlohmann/json.hpp"



#include "utils/inc/types.hpp"
#include "utils/inc/asserter.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

namespace kcc {
namespace kelf {

using json = nlohmann::json;

/***********************************************************************
***********************************************************************/
DmaDescription::DmaDescription()
{
}


/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::gBlockIdForQueue(const std::string& queName)
{
    kcc_int32 blockId;
    const auto it = m_QueueToBlockId.find(queName);
    if (it == m_QueueToBlockId.end()) {
        blockId = 0;
    } else {
        blockId = (*it).second;
    }
    m_QueueToBlockId[queName] = blockId + 1;
    return blockId;
}

/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::gNumBlockIdsForQueue(const std::string& queName) const
{
    const auto it = m_QueueToBlockId.find(queName);
    if (it == m_QueueToBlockId.end()) {
        return 0;
    } else {
        return (*it).second;
    }
}


/***********************************************************************
***********************************************************************/
auto
DmaDescription::startNewDmaBlockToTpb(EngineId engId, bool qWeights)
    -> DmaBlockToTpb&
{
    m_DmaBlocksToTpb.push_back(DmaBlockToTpb(*this, engId, qWeights));
    return m_DmaBlocksToTpb[m_DmaBlocksToTpb.size()-1];
}


/***********************************************************************
***********************************************************************/
auto
DmaDescription::startNewDmaBlockFromTpb(EngineId engId, bool qOut)
    -> DmaBlockFromTpb&
{
    m_DmaBlocksFromTpb.push_back(DmaBlockFromTpb(*this, engId, qOut));
    return m_DmaBlocksFromTpb[m_DmaBlocksFromTpb.size()-1];
}

/***********************************************************************
***********************************************************************/
auto
DmaDescription::startNewDmaBlockInput()
    -> DmaBlockInput&
{
    m_DmaBlocksInput.push_back(DmaBlockInput(*this));
    return m_DmaBlocksInput[ m_DmaBlocksInput.size()-1];
}



/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gJsonFileName(EngineId engId)
{
    switch (engId) {
    case EngineId::PeArray:
        return m_PeJsonFileName;
        break;
    case EngineId::Activation:
        return m_ActJsonFileName;
        break;
    case EngineId::Pooling:
        return m_PoolJsonFileName;
        break;
    default:
        Assert(false, "Wrong engine id ", static_cast<int>(engId));
        return nullptr;
        break;
    }
    return nullptr;
}

/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gEngineName(EngineId engId)
{
    switch (engId) {
    case EngineId::PeArray:
        return "pe";
        break;
    case EngineId::Activation:
        return "act";
        break;
    case EngineId::Pooling:
        return "pool";
        break;
    default:
        Assert(false, "Wrong engine id ", static_cast<int>(engId));
        return nullptr;
        break;
    }
    return nullptr;
}


/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gSymbolicStateBuffer()
{
    return "$SB";
}

/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gSymbolicOutput()
{
    return "$OUT";
}

/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gSymbolicInput()
{
    return "$IN";
}

/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gSymbolicInQueue()
{
    return "$IN_QUE";
}

/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gSymbolicOutQueue()
{
    return "$OUT_QUE";
}

/***********************************************************************
***********************************************************************/
auto DmaDescription::gFileSymbolicId(const std::string& fileName)
    -> FileIdType
{
    auto it = m_FileNameToId.find(fileName);
    if (m_FileNameToId.end() == it) {
        char buf[256];
        snprintf(buf, sizeof(buf)/sizeof(buf[0])-1, "$F%d", m_FileIdCnt++);
        FileIdType fileId(buf);
        m_FileNameToId[fileName] = fileId;
        return fileId;
    } else {
        return (*it).second;
    }
}

/***********************************************************************
***********************************************************************/
std::string
DmaDescription::gSymbolicQueue(EngineId engId, bool inpt, bool weight) const
{
    const char* engName = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engName = "pool";
        break;
    case EngineId::Activation:
        engName = "act";
        break;
    case EngineId::PeArray:
        engName = "pe";
        break;
    default:
        Assert(false, "Must be Pool, Act, or PeArray");
        engName = nullptr;
        break;
    }
    char buf[256];
    sprintf(buf, "$%s_%s", engName, inpt ? (weight ? "in_w" : "in_d") : "out");
    return std::string(buf);
}


/***********************************************************************
***********************************************************************/
void
DmaDescription::writeDmaDescriptors(
    const char* binFileName,
    EngineId engId)
{
    const char* const jsonFileName = gJsonFileName(engId);
    const char* name = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        name = "pool_json";
        break;
    case EngineId::Activation:
        name = "act_json";
        break;
    case EngineId::PeArray:
        name = "pe_array_json";
        break;
    default:
        Assert(false, "Must be Pool, Act, or PeArray");
        break;
    }

    json j;
    j["instr"] = binFileName;
    j["name"] = name;

    std::vector<json> jDmaBlocks;

    for (const auto& dmaBlockToTpb : m_DmaBlocksToTpb) {
        if (dmaBlockToTpb.gTriggerEngineId() != engId) {
            continue;
        }

        json jBlockToTpb;
        jBlockToTpb["queue"]    = dmaBlockToTpb.gQueueName();
        jBlockToTpb["id"]       = dmaBlockToTpb.gBlockId();
        jBlockToTpb["event"]    = dmaBlockToTpb.gEventId();

        std::vector<json> jDmaDescs;
        for (const auto& desc : dmaBlockToTpb.gDescs()) {
            json jDmaDesc;
            jDmaDesc["from"]         = desc.gSrcFileId();
            jDmaDesc["from_off"]     = desc.gSrcFileAddress();
            jDmaDesc["to"]           = gSymbolicStateBuffer();
            jDmaDesc["to_off"]       = desc.gDstSbAddress();
            jDmaDesc["size"]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jBlockToTpb["desc"] = jDmaDescs;

        jDmaBlocks.push_back(jBlockToTpb);
    }

    for (const auto& dmaBlockFromTpb : m_DmaBlocksFromTpb) {
        if (dmaBlockFromTpb.qOut()) {
            continue;
        }
        if (dmaBlockFromTpb.gTriggerEngineId() != engId) {
            continue;
        }

        json jBlockFromTpb;
        jBlockFromTpb["queue"]      = dmaBlockFromTpb.gQueueName();
        jBlockFromTpb["event"]      = dmaBlockFromTpb.gEventId();
        jBlockFromTpb["id"]         = dmaBlockFromTpb.gBlockId();

        std::vector<json> jDmaDescs;
        for (const auto& desc : dmaBlockFromTpb.gDescs()) {
            json jDmaDesc;
            jDmaDesc["from"]         = gSymbolicStateBuffer();
            jDmaDesc["from_off"]     = desc.gSrcSbAddress();
            jDmaDesc["to"]           = desc.gDstFileId();
            jDmaDesc["to_off"]       = desc.gDstFileAddress();
            jDmaDesc["size"]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jBlockFromTpb["desc"] = jDmaDescs;

        jDmaBlocks.push_back(jBlockFromTpb);
    }


    j["dma"] = jDmaBlocks;

    std::ofstream o(jsonFileName);
    o << std::setw(4) << j << std::endl;
}


/***********************************************************************
{
    "name" : "the test that does nothing",
    "var" : {
        "$W" : { "type":"file", "file_name":"weights.bin"},
        "$SB" : { "type": "state-buffer" },
        "$IN" : { "type": "io", "size": 1024 },
        "$OUT" : { "type": "io", "size": 1024 }
    },
    "dma_queue" : {
        "$IN_QID" : { "type": "in" },
        "$OUT_QID" : { "type": "out" },
        "$W_QID" : { "type" : "data", "owner": "pe" }
    },
    "pe" : "pe.json",
    "act" : "act.json",
    "pool" : "pool.json",
    "host" : "host.json"
}
***********************************************************************/
void
DmaDescription::writeDefinitions()
{
    std::array<EngineId, 3> engIds = { {EngineId::PeArray, EngineId::Pooling, EngineId::Activation} };
    json j;
    j["name"] = "definition";
    for (auto engId : engIds) {
        j[gEngineName(engId)] = gJsonFileName(engId);
    }
    j["host"] = m_HostJsonFileName;
    {
        json jDmaQueue;
        json queDesc;

        queDesc["type"] = "in";
        jDmaQueue[gSymbolicInQueue()] = queDesc;

        queDesc["type"] = "out";
        jDmaQueue[gSymbolicOutQueue()] = queDesc;

        for (auto engId : engIds) {
            std::string queName = gSymbolicQueue(engId, true, true);
            if (gNumBlockIdsForQueue(queName) > 0) {
                queDesc["type"] = "data";
                queDesc["owner"] = gEngineName(engId);
                jDmaQueue[queName]    = queDesc; // input for weights
            }

            queName = gSymbolicQueue(engId, true, false);
            if (gNumBlockIdsForQueue(queName) > 0) {
                queDesc["type"] = "data";
                queDesc["owner"] = gEngineName(engId);
                jDmaQueue[queName]   = queDesc; // input for data
            }

            queName = gSymbolicQueue(engId, false, false);
            if (gNumBlockIdsForQueue(queName) > 0) {
                queDesc["type"] = "data";
                queDesc["owner"] = gEngineName(engId);
                jDmaQueue[queName]  = queDesc; // output
            }
        }

        j["dma_queue"] = jDmaQueue;
    }
    {
        json jVars;
        {
            json varDesc;

            varDesc["type"] = "state-buffer";
            jVars["$SB"] = varDesc; 

            kcc_int64 numInBytes = 0;
            for (const auto& inBlock : m_DmaBlocksInput) {
                numInBytes += inBlock.size();
            }
            varDesc["type"] = "io";
            varDesc["size"] = numInBytes;
            jVars[gSymbolicInput()] = varDesc;

            kcc_int64 numOutBytes = 0;
            for (const auto& outBlock : m_DmaBlocksFromTpb) {
                if (outBlock.qOut()) {
                    numOutBytes += outBlock.size();
                }
            }
            varDesc["type"] = "io";
            varDesc["size"] = numOutBytes;
            jVars[gSymbolicOutput()] = varDesc;
        }
        {
            json varDesc;
            varDesc["type"] = "file";
            for (const auto& kv : m_FileNameToId) {
                //std::map<std::string, FileIdType>   
                varDesc["file_name"] = kv.first;
                jVars[kv.second] = varDesc;
            }
        }

        j["var"] = jVars;
    }

    std::ofstream o(m_DefJsonFileName);
    o << std::setw(4) << j << std::endl;
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::writeInOutDescriptors()
{
    json j;

    j["name"] = "host_json";
    std::vector<json> jDmaBlocks;

    for (const auto& dmaBlock : m_DmaBlocksInput) {
        json jDmaBlock;
        jDmaBlock["queue"]  = gSymbolicInQueue();
        jDmaBlock["event"]  = dmaBlock.gEventId();
        jDmaBlock["id"]     = dmaBlock.gBlockId();

        std::vector<json> jDmaDescs;
        for (const auto& desc : dmaBlock.gDescs()) {
            json jDmaDesc;
            jDmaDesc["from"]         = desc.gSrcFileId();
            jDmaDesc["from_off"]     = desc.gSrcFileAddress();
            jDmaDesc["to"]           = gSymbolicStateBuffer();
            jDmaDesc["to_off"]       = desc.gDstSbAddress();
            jDmaDesc["size"]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jDmaBlock["desc"] = jDmaDescs;
        jDmaBlocks.push_back(jDmaBlock);
    }


    for (const auto& dmaBlock : m_DmaBlocksFromTpb) {
        if (!dmaBlock.qOut()) {
            continue;
        }
        json jDmaBlock;
        jDmaBlock["queue"]  = gSymbolicOutQueue();
        // Output queues are polled.
        //jDmaBlock["event"]  = dmaBlock.gEventId();
        jDmaBlock["id"]     = dmaBlock.gBlockId();

        std::vector<json> jDmaDescs;
        for (const auto& desc : dmaBlock.gDescs()) {
            json jDmaDesc;
            jDmaDesc["from"]         = gSymbolicStateBuffer();
            jDmaDesc["from_off"]     = desc.gSrcSbAddress();
            jDmaDesc["to"]           = gSymbolicOutput();
            jDmaDesc["to_off"]       = desc.gDstFileAddress();
            jDmaDesc["size"]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jDmaBlock["desc"] = jDmaDescs;
        jDmaBlocks.push_back(jDmaBlock);
    }


    j["dma"] = jDmaBlocks;

    std::ofstream o(m_HostJsonFileName);
    o << std::setw(4) << j << std::endl;
}

/***********************************************************************
***********************************************************************/
kcc_uint64
DmaDescription::DmaBlockInput::size() const
{
    kcc_uint64 numInBytes = 0;
    for (const auto& desc : m_Descs) {
        numInBytes += desc.gNumBytes();
    }
    return numInBytes;
}

/***********************************************************************
***********************************************************************/
kcc_uint64
DmaDescription::DmaBlockFromTpb::size() const
{
    kcc_uint64 numBytes = 0;
    for (const auto& desc : m_Descs) {
        numBytes += desc.gNumBytes();
    }
    return numBytes;
}

/***********************************************************************
***********************************************************************/
kcc_uint64
DmaDescription::DmaBlockToTpb::size() const
{
    kcc_uint64 numBytes = 0;
    for (const auto& desc : m_Descs) {
        numBytes += desc.gNumBytes();
    }
    return numBytes;
}

}}

