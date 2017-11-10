#ifndef ARB_PREC_H
#define ARB_PREC_H

#include "fp16/fp16.h"
#include <cstdint>
#include <typeinfo>
#include <limits.h>
#include <assert.h>
#include <iostream>
#include <inttypes.h>

//#include "types.h"
//#include "isa.h"

typedef union ArbPrecData {
    uint8_t  uint8;
    uint8_t  uint8_tuple[2];
    uint16_t uint16;
    uint32_t uint32;
    uint32_t uint32_tuple[2];
    uint64_t uint64;
    int8_t   int8;
    int8_t   int8_tuple[2];
    int16_t  int16;
    int32_t  int32;
    int32_t  int32_tuple[2];
    int64_t  int64;
    uint16_t fp16; /* tmp! */
    float    fp32;
    unsigned char ch[8];
    uint64_t raw;
    ArbPrecData() {raw = 0;}
} ArbPrecData;

namespace ArbPrec
{
    // TODO: find compile time log 2

/** Multiply two arbitrary precision values.
 *
 *  Performs an 'x*y' on two arbitrary precision values.
 *
 *  @param[in] x
 *  @param[in] y
 *  @param[in] in_type - The type of x and y.
 *  @param[out] out_type - The type of the result of 'x*y'.
 *
 *  @return x*y
 */
inline ArbPrecData multiply(const ArbPrecData& x, const ArbPrecData& y,
                            ARBPRECTYPE in_type, ARBPRECTYPE& out_type);

/** Add two arbitrary precision values.
 *
 *  Performs an 'x+y' on two arbitrary precision values.
 *
 *  @param[in] x
 *  @param[in] y
 *  @param[in] in_type - The type of x and y.
 *
 *  @return x+y
 */
inline ArbPrecData add(const ArbPrecData& x, const ArbPrecData& y,
                       ARBPRECTYPE in_type);

/** Divides an arbitrary precision values by an unsigned integer.
 *
 *  Performs 'x/uy'.
 *
 *  @param[in] x - Arbitrary precision as numerator.
 *  @param[in] uy - signed integer as denominator.
 *  @param[in] in_type - The type of x.
 *
 *  @return x/uy
 */
inline ArbPrecData int_divide(const ArbPrecData& x, int y,
                               ARBPRECTYPE in_type);


/** Compares two arbitrary precision values.
 *
 *  Performs an 'x<y' on two arbitrary precision values.
 *
 *  @param[in] x
 *  @param[in] y
 *  @param[in] in_type - The type of x and y.
 *
 *  @return x<y as boolean.
 */
inline bool gt(ArbPrecData& x, ArbPrecData& y, ARBPRECTYPE in_type);
/** Converts to fp32 to input type
 *
 *  Performs a casting of arbitrary precission float32 value x to out_type
 *
 *  @param[in] x - the value to convert.
 *  @param[in] in_type - The type of x.
 *
 *  @return (fp32_t) copy(x)
 */
inline ArbPrecData cast_to_fp32(const ArbPrecData& x, ARBPRECTYPE in_type);

/** Converts from fp32 to input type
 *
 *  Performs an (out_type)x on arbitrary precision float32 value.
 *
 *  @param[in] x - the value to convert.
 *  @param[in] out_type - The output type.
 *
 *  @return (out_type) copy(x)
 */
inline ArbPrecData cast_from_fp32(const ArbPrecData& x, ARBPRECTYPE out_type);


/** Write an arbitrary precision value to a file.
 *
 *  @param[in] f - The file to write to.
 *  @param[in] x - The value to write.
 *  @param[in] type - The type of x.
 */
inline void dump(FILE* f, const ArbPrecData& x, ARBPRECTYPE type);

namespace details
{

/** Template to convert from ARBPRECTYPE enum values to C++ types.
 *
 *  ex. true == std::is_same<TypeOf<ARBPRECTYPE::UINT16>, uint16_t>
 */
template <int Type> struct TypeOf { using type = bool; };
template <> struct TypeOf<ARBPRECTYPE::INT8> { using type = int8_t; };
template <> struct TypeOf<ARBPRECTYPE::UINT8> { using type = uint8_t; };
template <> struct TypeOf<ARBPRECTYPE::INT16> { using type = int16_t; };
template <> struct TypeOf<ARBPRECTYPE::UINT16> { using type = uint16_t; };
template <> struct TypeOf<ARBPRECTYPE::INT32> { using type = int32_t; };
template <> struct TypeOf<ARBPRECTYPE::UINT32> { using type = uint32_t; };
template <> struct TypeOf<ARBPRECTYPE::INT64> { using type = int64_t; };
template <> struct TypeOf<ARBPRECTYPE::UINT64> { using type = uint64_t; };
template <> struct TypeOf<ARBPRECTYPE::FP16> { using type = float; };
template <> struct TypeOf<ARBPRECTYPE::FP32> { using type = float; };

/** Template to extract the ARBPRECTYPE value from a ArbPrecData union.
 *
 *  ex. Extract<INT8>::extract returns a reference to ArbPrecData::int8.
 */
template <int Type> struct Extract
{
    static inline typename TypeOf<Type>::type& extract(const ArbPrecData&)
    {
        assert(0 && "unsupported type" && Type);

        static typename TypeOf<Type>::type unused;
        return unused;
    }
};

/** Template to extract a specific member from an ArbPrecData union.
 *
 *  @tparam M - The type of the member.
 *  @tparam Member - The pointer of the member.
 *
 *  This class is intended to be used as a inherited class of a struct, which
 *  is itself a specialization of the Extract template.
 *
 *  This says that Extract of INT8 returns the member of the union int8:
 *  template <> struct Extract<INT8> : ExtractMember<decltype(&::int8), &::int8>
 */
template <typename M, M Member>
struct ExtractMember
{
    // Utility template to get the underlying type of a member pointer,
    // since '&Struct::Member' is of type 'MemberType Struct::*'
    template <typename T> struct UnderlyingMemberType { };
    template <typename T, typename S> struct UnderlyingMemberType<T S::*>
    {
        using type = T;
    };

    using member = typename UnderlyingMemberType<M>::type;
    using const_member = typename std::add_const<member>::type;

    /** Extract a reference to the member.
     *
     *  This returns a reference-to-const or a reference, depending on if
     *  the passed parameter is a const ArbPrecData or a ArbPrecData.
     */
    template <typename T>
    static inline auto extract(T& v) ->
        typename std::conditional<std::is_const<T>::value,
                                  const_member&, member&>::type
    {
        static_assert(std::is_same<ArbPrecData,
                                   typename std::remove_cv<T>::type>::value,
                      "Parameter type must be ArbPrecData");

        return v.*Member;
    }
};


template <> struct Extract<ARBPRECTYPE::INT8> :
        ExtractMember<decltype(&ArbPrecData::int8), &ArbPrecData::int8> {};
template <> struct Extract<ARBPRECTYPE::UINT8> :
        ExtractMember<decltype(&ArbPrecData::uint8), &ArbPrecData::uint8> {};
template <> struct Extract<ARBPRECTYPE::INT16> :
        ExtractMember<decltype(&ArbPrecData::int16), &ArbPrecData::int16> {};
template <> struct Extract<ARBPRECTYPE::UINT16> :
        ExtractMember<decltype(&ArbPrecData::uint16), &ArbPrecData::uint16> {};
template <> struct Extract<ARBPRECTYPE::INT32> :
        ExtractMember<decltype(&ArbPrecData::int32), &ArbPrecData::int32> {};
template <> struct Extract<ARBPRECTYPE::UINT32> :
        ExtractMember<decltype(&ArbPrecData::uint32), &ArbPrecData::uint32> {};
template <> struct Extract<ARBPRECTYPE::INT64> :
        ExtractMember<decltype(&ArbPrecData::int64), &ArbPrecData::int64> {};
template <> struct Extract<ARBPRECTYPE::UINT64> :
        ExtractMember<decltype(&ArbPrecData::uint64), &ArbPrecData::uint64> {};
template <> struct Extract<ARBPRECTYPE::FP16> :
        ExtractMember<decltype(&ArbPrecData::fp16), &ArbPrecData::fp16> {};
template <> struct Extract<ARBPRECTYPE::FP32> :
        ExtractMember<decltype(&ArbPrecData::fp32), &ArbPrecData::fp32> {};

/** Syntactic helper function for Extract<T>::extract.
 *
 *  Allows you to write extract<INT8>(value) instead of
 *  Extract<INT8>::extract(value)
 *
 *  @param[in] v - Const or non-const reference to an ArbPrecData.
 *  @return - A reference to the underlying primitive type from v with the
 *            same const-ness as v.
 */
template<int Type, typename T>
inline auto extract(T& v) ->
        decltype(Extract<Type>::extract(v))
{
    return Extract<Type>::extract(v);
}

/** Template to iterate through all ARBPRECTYPEs and call an appropriate
 *  operator for the given type.
 *
 *  @tparam Op - A class with an 'eval' template function which will
 *               perform an operation for each ARBPRECTYPE value.
 *  @tparam Type - The type this Unroll instance handles.
 *  @tparam Args - The arguments to 'eval'.
 *
 *  This template is typically instantiated via 'unroll'.  It can be
 *  imagined as a compile-time generated 'case' statement.  If the 'Type'
 *  of this template instance matches the 'type' passed into the 'unroll'
 *  function, then the 'Op::eval' operator is called.  Otherwise, the
 *  template calls itself with Type as 'Type-1'.
 */
template <typename Op, int Type, typename... Args>
struct Unroll
{
    /** Calls 'eval' if Type == value_type, otherwise calls next 'unroll'.
     *
     *  @param[in] value_type - The "input" type.
     *  @param[out] result_type - The "output" type.
     *  @param args - The arguments to call 'eval' with.
     *
     *  @return Whatever 'Op::eval' returns.
     *
     *  Implementation notes:
     *
     *  C++14 and above support 'automatic return type deduction' but
     *  C++11 does not, so we have to define the return type of this
     *  function.  Using the 'auto foo() -> return-type' syntax so that
     *  the type deduction using 'decltype' clutters less.
     *
     *  Since 'Args' may be passed by value, or by l-value or r-value
     *  references, use std::forward to ensure they are forwarded as the
     *  appropriate type.
     */
    static inline auto unroll(
            ARBPRECTYPE value_type, ARBPRECTYPE& result_type,
            Args ...args) ->
        decltype(Op::template eval<Type>(
                    std::forward<Args>(args)..., result_type))
    {
        if (Type == value_type)
        {
            return Op::template eval<Type>(
                    std::forward<Args>(args)..., result_type);
        }
        else
        {
            return Unroll<Op, Type-1, Args...>::unroll(
                    value_type, result_type, std::forward<Args>(args)...);
        }
    }
};

/** Template specialization for the invalid / 0 type to stop recursion. */
template <typename Op, typename... Args>
struct Unroll<Op, ARBPRECTYPE::INVALID_ARBPRECTYPE, Args...>
{
    static inline auto unroll(
            ARBPRECTYPE, ARBPRECTYPE& result_type,
            Args ...args __attribute__((unused))) ->
        decltype(Op::template eval<ARBPRECTYPE::INVALID_ARBPRECTYPE>(
                std::forward<Args>(args)..., result_type))
    {
        assert(0 && "unsupported type");
    }
};

/** Template function to call 'Op::eval<Type>' on the appropriate type.
 *
 *  @param[in] value_type - The 'input type'.
 *  @param[out] result_type - The 'output type'.
 *  @param args - The arguments to call 'Op::eval' with.
 */
template <typename Op, typename... Args>
inline auto unroll(
        ARBPRECTYPE value_type, ARBPRECTYPE& result_type,
        Args ...args) ->
    decltype(Unroll<Op, ARBPRECTYPE::NUM_ARBPRECTYPE-1, Args...>::unroll(
            value_type, result_type, std::forward<Args>(args)...))
{
    return Unroll<Op, ARBPRECTYPE::NUM_ARBPRECTYPE-1, Args...>::unroll(
            value_type, result_type, std::forward<Args>(args)...);
}

/** Template to determine the result type of x*y.
 *
 * Some multiplications cause type conversions:
 *      ex. INT8 * INT8 = INT32
 */
template <int Type>
constexpr ARBPRECTYPE multResult() { return ARBPRECTYPE(Type); }
template <> constexpr ARBPRECTYPE multResult<INT8>() { return INT32; }
template <> constexpr ARBPRECTYPE multResult<UINT8>() { return UINT32; }
template <> constexpr ARBPRECTYPE multResult<INT16>() { return INT32; }
template <> constexpr ARBPRECTYPE multResult<UINT16>() { return UINT32; }
template <> constexpr ARBPRECTYPE multResult<FP16>() { return FP32; }

/** Operator class for 'unroll' that will perform multiplications. */
struct Multiply
{
    template <int Type>
    static inline ArbPrecData eval(
            const ArbPrecData& x, const ArbPrecData& y, ARBPRECTYPE& r)
    {
        static constexpr auto result_type = multResult<Type>();
        using result_t = typename TypeOf<result_type>::type;

        r = result_type;
        result_t result =
            static_cast<result_t>(extract<Type>(x)) *
            static_cast<result_t>(extract<Type>(y));

        ArbPrecData real_result;
        Extract<result_type>::extract(real_result) = result;

        return real_result;
    }
};

template <>
inline ArbPrecData Multiply::eval<ARBPRECTYPE::FP16>(
        const ArbPrecData& x, const ArbPrecData& y, ARBPRECTYPE& r)
{
    static constexpr auto Type = ARBPRECTYPE::FP16;
    static constexpr auto result_type = multResult<Type>();
    using result_t = typename TypeOf<result_type>::type;

    r = result_type;
    result_t result = 
            fp16_ieee_to_fp32_value(extract<Type>(x)) *
            fp16_ieee_to_fp32_value(extract<Type>(y));

    ArbPrecData real_result;
    Extract<result_type>::extract(real_result) = result;

    return real_result;
}

/** Operator class for 'unroll' that will perform additions. */
struct Add
{
    template <int Type>
    static inline ArbPrecData eval(
            const ArbPrecData& x, const ArbPrecData& y, ARBPRECTYPE& r)
    {
        static constexpr auto result_type = ARBPRECTYPE(Type);
        using result_t = typename TypeOf<result_type>::type;

        r = result_type;
        result_t result =
            static_cast<result_t>(extract<Type>(x)) +
            static_cast<result_t>(extract<Type>(y));

        ArbPrecData real_result;
        Extract<result_type>::extract(real_result) = result;

        return real_result;
    }
};

template <>
inline ArbPrecData Add::eval<ARBPRECTYPE::FP16>(
        const ArbPrecData& x, const ArbPrecData& y, ARBPRECTYPE& r)
{
    static constexpr auto Type = ARBPRECTYPE::FP16;
    static constexpr auto result_type = Type;
    using result_t = typename TypeOf<result_type>::type;

    r = result_type;
    result_t result = fp16_ieee_from_fp32_value(
        fp16_ieee_to_fp32_value(extract<Type>(x)) +
        fp16_ieee_to_fp32_value(extract<Type>(y)));

    ArbPrecData real_result;
    Extract<result_type>::extract(real_result) = result;

    return real_result;
}

/** Operator class for 'unroll' that will perform division with signed. */
struct IntDivide
{
    template <int Type>
    static inline ArbPrecData eval(
            const ArbPrecData& x, int y, ARBPRECTYPE& r)
    {
        static constexpr auto result_type = ARBPRECTYPE(Type);
        using result_t = typename TypeOf<result_type>::type;

        r = result_type;
        result_t result = static_cast<result_t>(extract<Type>(x)) / y;

        ArbPrecData real_result;
        Extract<result_type>::extract(real_result) = result;

        return real_result;
    }
};

template <>
inline ArbPrecData IntDivide::eval<ARBPRECTYPE::FP16>(
        const ArbPrecData& x, int y, ARBPRECTYPE& r)
{
    static constexpr auto Type = ARBPRECTYPE::FP16;
    static constexpr auto result_type = Type;
    using result_t = typename TypeOf<result_type>::type;

    r = result_type;
    result_t result = fp16_ieee_from_fp32_value(
        fp16_ieee_to_fp32_value(extract<Type>(x)) / y);

    ArbPrecData real_result;
    Extract<result_type>::extract(real_result) = result;

    return real_result;
}

/** Operator class for 'unroll' that will perform '>'. */
struct GreaterThan
{
    template <int Type>
    static inline bool eval(
            const ArbPrecData& x, const ArbPrecData& y, ARBPRECTYPE&)
    {
        return extract<Type>(x) > extract<Type>(y);
    }
};

template <>
inline bool GreaterThan::eval<ARBPRECTYPE::FP16>(
        const ArbPrecData& x, const ArbPrecData& y, ARBPRECTYPE&)
{
    static constexpr auto Type = ARBPRECTYPE::FP16;

    return fp16_ieee_to_fp32_value(extract<Type>(x)) >
        fp16_ieee_to_fp32_value(extract<Type>(y));
}

/** Operator class for 'unroll' that will perform conversion from fp32 */
struct CastToFp32
{
    template <int InType>
    static inline ArbPrecData eval(const ArbPrecData& x, ARBPRECTYPE&)
    {
        static constexpr auto OutType = ARBPRECTYPE::FP32;
        using result_t = typename TypeOf<OutType>::type;
        result_t result = static_cast<result_t>(extract<InType>(x));

        ArbPrecData real_result;
        Extract<OutType>::extract(real_result) = result;

        return real_result;
    }
};

template <>
inline ArbPrecData CastToFp32::eval<ARBPRECTYPE::FP16>(
        const ArbPrecData& x, ARBPRECTYPE&)
{
    static constexpr auto InType = ARBPRECTYPE::FP16;
    static constexpr auto OutType = ARBPRECTYPE::FP32;
    using result_t = typename TypeOf<OutType>::type;
    result_t result = fp16_ieee_to_fp32_value(extract<InType>(x));

    ArbPrecData real_result;
    Extract<OutType>::extract(real_result) = result;

    return real_result;
}

/** Operator class for 'unroll' that will perform conversion  to fp32*/
struct CastFromFp32
{
    template <int OutType>
    static inline ArbPrecData eval(const ArbPrecData& x, ARBPRECTYPE&)
    {
        static constexpr auto InType = ARBPRECTYPE::FP32;
        using result_t = typename TypeOf<OutType>::type;
        result_t result = static_cast<result_t>(extract<InType>(x));

        ArbPrecData real_result;
        Extract<OutType>::extract(real_result) = result;

        return real_result;
    }
}
;
template <>
inline ArbPrecData CastFromFp32::eval<ARBPRECTYPE::FP16>(
        const ArbPrecData& x, ARBPRECTYPE&)
{
    static constexpr auto InType = ARBPRECTYPE::FP32;
    static constexpr auto OutType = ARBPRECTYPE::FP16;

    using result_t = typename TypeOf<OutType>::type;
    result_t result = fp16_ieee_from_fp32_value(extract<InType>(x));

    ArbPrecData real_result;
    Extract<OutType>::extract(real_result) = result;

    return real_result;
}

/** Operator class for 'unroll' that will print a value. */
struct Dump
{
    template <int Type>
    static inline void eval(FILE* f, const ArbPrecData& x, ARBPRECTYPE&)
    {
        constexpr bool is_fp =
            std::is_floating_point<typename TypeOf<Type>::type>::value;
        constexpr bool is_integer =
            std::is_integral<typename TypeOf<Type>::type>::value;

        assert(is_fp || is_integer);

        if (is_fp)
        {
            fprintf(f, "%f", static_cast<double>(extract<Type>(x)));
        }
        else
        {
            fprintf(f, "%" PRId64, static_cast<int64_t>(extract<Type>(x)));
        }
    }
};

template <>
inline void Dump::eval<ARBPRECTYPE::FP16>(
        FILE* f, const ArbPrecData& x, ARBPRECTYPE&)
{
    fprintf(f, "%f", fp16_ieee_to_fp32_value(extract<ARBPRECTYPE::FP16>(x)));
}

} // details namespace

// ---- Implementations of ArbPrec functions forward-declared above. ----

/* multiply */
inline ArbPrecData multiply(const ArbPrecData& x, const ArbPrecData& y,
                            ARBPRECTYPE in_type, ARBPRECTYPE& out_type)
{
    return details::unroll<details::Multiply>(in_type, out_type, x, y);
}

/* add */
inline ArbPrecData add(const ArbPrecData& x, const ArbPrecData& y,
                       ARBPRECTYPE in_type)
{
    return details::unroll<details::Add>(in_type, in_type, x, y);
}

/* int_divide */
inline ArbPrecData int_divide(const ArbPrecData& x, int y,
                               ARBPRECTYPE in_type)
{
    return details::unroll<details::IntDivide>(in_type, in_type, x, y);
}

/* gt */
inline bool gt(ArbPrecData& x, ArbPrecData& y, ARBPRECTYPE in_type)
{
    return details::unroll<details::GreaterThan>(in_type, in_type, x, y);
}

/* cast_to_fp32 */
inline ArbPrecData cast_to_fp32(const ArbPrecData& x, 
        ARBPRECTYPE in_type)
{
    ARBPRECTYPE out_type = ARBPRECTYPE::FP32;
    return details::unroll<details::CastToFp32>(in_type, out_type, x);
}

/* cast_from_fp32 */
inline ArbPrecData cast_from_fp32(const ArbPrecData& x, 
        ARBPRECTYPE out_type)
{
    ARBPRECTYPE in_type = ARBPRECTYPE::FP32 ;
    return details::unroll<details::CastFromFp32>(out_type, in_type, x);
}


/* dump */
inline void dump(FILE* f, const ArbPrecData& x, ARBPRECTYPE type)
{
    return details::unroll<details::Dump>(type, type, f, x);
}

} // namespace ArbPrec

#endif // ARB_PREC_H
