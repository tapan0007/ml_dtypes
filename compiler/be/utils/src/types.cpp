#include <assert.h>

#include "utils/inc/types.hpp"


namespace kcc {
namespace utils {

const std::string&
poolType2Str(PoolType poolType)
{
    static const std::string maxPool(LayerTypeStr_MaxPool);
    static const std::string avgPool(LayerTypeStr_AvgPool);

    switch(poolType) {
    case PoolType::Max:
        return maxPool;
        break;
    case PoolType::Avg:
        return avgPool;
        break;
    default:
        assert(false && "Wrong Pool Type");
        break;
    }
    assert(false && "Wrong Pool Type");
    return maxPool;
}

PoolType
poolTypeStr2Id(const std::string& str)
{
    if (str == LayerTypeStr_MaxPool) {
        return PoolType::Max;
    } else if (str == LayerTypeStr_AvgPool) {
        return PoolType::Avg;
    } else {
        assert(false && "Wrong Pool Name");
    }
    assert(false && "Wrong Pool Name");
    return PoolType::None;
}

}}

