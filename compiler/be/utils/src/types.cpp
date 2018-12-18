#include <assert.h>
#include <cstring>

#include "utils/inc/asserter.hpp"
#include "utils/inc/types.hpp"


namespace kcc {
namespace utils {


const std::string&
poolType2Str(PoolType poolType)
{
    static const std::string maxPool(PoolTypeStr::MaxPool);
    static const std::string avgPool(PoolTypeStr::AvgPool);
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
    if (str == PoolTypeStr::MaxPool) {
        return PoolType::Max;
    } else if (str == PoolTypeStr::AvgPool) {
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
    if (str == EngineIdStr::PeArray) {
        return EngineId::PeArray;
    } else if (str == EngineIdStr::Activation) {
        return EngineId::Activation;
    } else if (str == EngineIdStr::Pool) {
        return EngineId::Pooling;
    } else if (str == EngineIdStr::StreamProc) {
        return EngineId::StreamProc;
    } else if (str == EngineIdStr::Angel) {
        return EngineId::AngelEng;
    } else {
        Assert(false, "Wrong Engine name ", str);
    }
    Assert(false, "Wrong Engine name ", str);
    return EngineId::None;
}

const std::string&
engineId2Str(EngineId engId)
{
    static const std::string peArrayEng(EngineIdStr::PeArray);
    static const std::string actEng(EngineIdStr::Activation);
    static const std::string poolEng(EngineIdStr::Pool);
    static const std::string spEng(EngineIdStr::StreamProc);
    static const std::string dmaEng(EngineIdStr::Dma);
    static const std::string angelEng(EngineIdStr::Angel);
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

    case EngineId::AngelEng:
        return angelEng;
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

const std::string& ActivationFunc2Str(ActivationFunc actFunc)
{
    static const std::string InvalidStr("INVALID");
    static const std::string IdentityStr("IDENTITY");
    static const std::string ReluStr("RELU");
    static const std::string LeakyReluStr("LEAKY_RELU");
    static const std::string PReluStr("PARAMETRIC_RELU");
    static const std::string SigmoidStr("SIGMOID");
    static const std::string TanhStr("TANH");
    static const std::string ExpStr("EXP");
    static const std::string SqrtStr("SQRT");
    static const std::string SoftplusStr("SOFTPLUS");

    switch (actFunc) {
    case ActivationFunc::Identity:
        return IdentityStr;
    case ActivationFunc::Relu:
        return ReluStr;
    case ActivationFunc::LeakyRelu:
        return LeakyReluStr;
    case ActivationFunc::PRelu:
        return PReluStr;
    case ActivationFunc::Sigmoid:
        return SigmoidStr;
    case ActivationFunc::Tanh:
        return TanhStr;
    case ActivationFunc::Exp:
        return ExpStr;
    case ActivationFunc::Sqrt:
        return SqrtStr;
    case ActivationFunc::Softplus:
        return SoftplusStr;
    default:
        return InvalidStr;
    }
}

 
#undef ELSEIF
#define ELSEIF(X)  } else if (0 == strcmp(tensorOp, TensorAluTypeStr::X)) { return TensorAluOpType::X;

TensorAluOpType
gAluOpType(const char* tensorOp)
{
    if (0 == strcmp(tensorOp, TensorAluTypeStr::Bypass)) {
        return TensorAluOpType::Bypass;
    ELSEIF(Add)
    ELSEIF(Sub)
    ELSEIF(Mult)
    ELSEIF(Div)
    ELSEIF(Max)
    ELSEIF(Min)
    ELSEIF(BwNot)
    ELSEIF(BwAnd)
    ELSEIF(BwOr)
    ELSEIF(BwXor)
    ELSEIF(LogAnd)
    ELSEIF(LogOr)
    ELSEIF(LogXor)
    ELSEIF(LogShiftLeft)
    ELSEIF(LogShiftRight)
    ELSEIF(ArithShiftLeft)
    ELSEIF(ArithShiftRight)
    ELSEIF(Equal)
    ELSEIF(Gt)
    ELSEIF(Ge)
    ELSEIF(Lt)
    ELSEIF(Le)
    } else {
        Assert(false, "Wrong Tensor Alu Opcode '", tensorOp, "'");
    }
    return TensorAluOpType::Le;
}
#undef ELSEIF

#undef CASE
#define CASE(X) case TensorAluOpType::X: return TensorAluTypeStr::X; break;

const char*
gAluOpTypeStr(TensorAluOpType opType)
{
    switch (opType) {
    CASE(Bypass)
    CASE(Add)
    CASE(Sub)
    CASE(Mult)
    CASE(Div)
    CASE(Max)
    CASE(Min)
    CASE(BwNot)
    CASE(BwAnd)
    CASE(BwOr)
    CASE(BwXor)
    CASE(LogAnd)
    CASE(LogOr)
    CASE(LogXor)
    CASE(LogShiftLeft)
    CASE(LogShiftRight)
    CASE(ArithShiftLeft)
    CASE(ArithShiftRight)
    CASE(Equal)
    CASE(Gt)
    CASE(Ge)
    CASE(Lt)
    CASE(Le)
    default:
        break;
        Assert(false, "Wrong Tensor Alu Opcode ", static_cast<kcc_int32>(opType));
    }
    return TensorAluTypeStr::Le;
}
#undef CASE

}}

