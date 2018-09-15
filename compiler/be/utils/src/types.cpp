#include <assert.h>

#include "utils/inc/asserter.hpp"
#include "utils/inc/types.hpp"

#include "layers/inc/layerconsts.hpp"

namespace kcc {
namespace utils {

const std::string&
poolType2Str(PoolType poolType)
{
    static const std::string maxPool(layers::LayerTypeStr_MaxPool);
    static const std::string avgPool(layers::LayerTypeStr_AvgPool);
    static const std::string badPool("Bad pool type");

    switch(poolType) {
    case PoolType::Max:
        return maxPool;
        break;
    case PoolType::Avg:
        return avgPool;
        break;
    default:
        Assert(false, "Wrong Pool Type", static_cast<int>(poolType));
        break;
    }
    Assert(false, "Wrong Pool Type", static_cast<int>(poolType));
    return badPool;
}

PoolType
poolTypeStr2Id(const std::string& str)
{
    if (str == layers::LayerTypeStr_MaxPool) {
        return PoolType::Max;
    } else if (str == layers::LayerTypeStr_AvgPool) {
        return PoolType::Avg;
    } else {
        Assert(false, "Wrong Pool Name", str);
    }
    Assert(false, "Wrong Pool Name", str);
    return PoolType::None;
}


EngineId
engineId2Str(const std::string& str)
{
    if (str == EngineIdStr_PeArray) {
        return EngineId::PeArray;
    } else if (str == EngineIdStr_Activation) {
        return EngineId::Activation;
    } else if (str == EngineIdStr_Pool) {
        return EngineId::Pooling;
    } else if (str == EngineIdStr_StreamProc) {
        return EngineId::StreamProc;
    } else if (str == EngineIdStr_Dma) {
        return EngineId::DmaEng;
    } else {
        Assert(false, "Wrong Engine name ", str);
    }
    Assert(false, "Wrong Engine name ", str);
    return EngineId::None;
}

const std::string&
engineId2Str(EngineId engId)
{
    static const std::string peArrayEng(EngineIdStr_PeArray);
    static const std::string actEng(EngineIdStr_Activation);
    static const std::string poolEng(EngineIdStr_Pool);
    static const std::string spEng(EngineIdStr_StreamProc);
    static const std::string dmaEng(EngineIdStr_Dma);
    static const std::string badEng("Bad Engine");

    switch(engId) {
    case EngineId::PeArray:
        return peArrayEng;
        break;
    case EngineId::Pooling:
        return poolEng;
        break;
    case EngineId::Activation:
        return actEng;
        break;

    case EngineId::DmaEng:
        return dmaEng;
        break;
    case EngineId::StreamProc:
        return spEng;
        break;
    default:
        Assert(false, "Wrong Engine ID ", static_cast<int>(engId));
    }
    Assert(false, "Wrong Engine ID ", static_cast<int>(engId));
    return badEng;
}

}}

