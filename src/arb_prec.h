#ifndef ARB_PREC_H
#define ARB_PREC_H

#include <cstdint>
#include <typeinfo>
#include <limits.h>
#include <assert.h>
#include <iostream>
#include "types.h"

size_t
sizeofArbPrecType(ARBPRECTYPE type);

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
    float    fp16;
    float    fp32;
    uint64_t raw;
} __attribute__ ((__packed__)) ArbPrecData;

#define EXTRACT(C, V) \
    (*((C *)V))


class ArbPrec
{
    // TODO: find compile time log 2
    public:
        /* multiply */
        static ArbPrecData multiply(ArbPrecData &x, ArbPrecData &y, 
                ARBPRECTYPE in_type, ARBPRECTYPE &out_type) {
            return _multiply(element_ptr(x, in_type), element_ptr(y, in_type), 
                    in_type, out_type);
        }
        static ArbPrecData multiply(void *x, void *y, ARBPRECTYPE in_type, 
                ARBPRECTYPE &out_type) {
            return _multiply(x, y, in_type, out_type);
        }

        /* add */
        static ArbPrecData add(ArbPrecData &x, ArbPrecData &y,
                ARBPRECTYPE in_type) {
            return _add(element_ptr(x, in_type), element_ptr(y, in_type), 
                    in_type);
        }
        static ArbPrecData add(void *x, void *y, ARBPRECTYPE in_type) {
            return _add(x, y, in_type);
        }

        /* uint_divide */
        static ArbPrecData uint_divide(ArbPrecData &x, unsigned int uy,
                ARBPRECTYPE in_type) {
            return _uint_divide(element_ptr(x, in_type), uy, in_type);
        }
        static ArbPrecData uint_divide(void *x, unsigned int uy, 
                ARBPRECTYPE in_type) {
            return _uint_divide(x, uy, in_type);
        }

        /* gt */
        static bool gt(ArbPrecData &x, ArbPrecData &y,
                ARBPRECTYPE in_type) {
            return _gt(element_ptr(x, in_type), element_ptr(y, in_type), 
                    in_type);
        }
        static bool gt(void *x, void *y, ARBPRECTYPE in_type) {
            return _gt(x, y, in_type);
        }

        /* dump */
        static void dump(FILE *f, ArbPrecData &x, ARBPRECTYPE type) {
            _dump(f, element_ptr(x, type), type);
        }
        static void dump(FILE *f, void *x, ARBPRECTYPE type) {
            _dump(f, x, type);
        }

        static void *element_ptr(ArbPrecData &x, ARBPRECTYPE type) {
            void *ptr = NULL;
            switch (type) {
                case UINT8:
                    ptr = &(x.uint8);
                    break;
                case INT8:
                    ptr = &(x.int8);
                    break;
                case UINT16:
                    ptr = &(x.uint16);
                    break;
                case INT16:
                    ptr = &(x.int16);
                    break;
                case UINT32:
                    ptr = &(x.uint32);
                    break;
                case INT32:
                    ptr = &(x.int32);
                    break;
                case FP16:
                    ptr = &(x.fp16);
                    break;
                case FP32:
                    ptr = &(x.fp32);
                    break;
                default:
                    assert(0);
            }
            return ptr;
        }
    private:
        ArbPrec()  {};
        ~ArbPrec() {}
        static ArbPrecData _multiply(void *x, void *y, ARBPRECTYPE in_type, 
                ARBPRECTYPE &out_type) {
            ArbPrecData ap;
            if (in_type == UINT8) {
                out_type = UINT32;
                ap.uint32 = uint32_t(EXTRACT(uint8_t, x)) * 
                    uint32_t(EXTRACT(uint8_t, y));
            } else if (in_type == INT8) {
                out_type = INT32;
                ap.uint32 = int32_t(EXTRACT(int8_t, x)) * 
                    int32_t(EXTRACT(int8_t, y));
            } else if (in_type == UINT16) {
                out_type = UINT32;
                ap.uint32 = uint32_t(EXTRACT(uint16_t, x)) * 
                    uint32_t(EXTRACT(uint16_t, y));
            } else if (in_type == INT16) {
                out_type = INT32;
                ap.int32 = int32_t(EXTRACT(int16_t, x)) * 
                    int32_t(EXTRACT(int16_t, y));
            } else if (in_type == UINT32) {
                out_type = UINT32;
                ap.uint32 = uint32_t(EXTRACT(uint32_t, x)) * 
                    uint32_t(EXTRACT(uint32_t, y));
            } else if (in_type == INT32) {
                out_type = INT32;
                ap.int32 = int32_t(EXTRACT(int32_t, x)) * 
                    int32_t(EXTRACT(int32_t, y));
            } else {
                assert(0 && "unsupported combo");
            }
            return ap;
        }

        static ArbPrecData _add(void *x, void *y, ARBPRECTYPE in_type) {
            ArbPrecData ap;
            if (in_type == UINT32) {
                ap.uint32 = uint32_t(EXTRACT(uint32_t, x)) + 
                    uint32_t(EXTRACT(uint32_t, y));
            } else if (in_type == UINT32) {
                ap.int32 = int32_t(EXTRACT(int32_t, x)) + 
                    int32_t(EXTRACT(int32_t, y));
            } else if (in_type == FP32) {
                ap.fp32 = float(EXTRACT(float, x)) + 
                    float(EXTRACT(float, y));
            }
            return ap;
        }

        static ArbPrecData _uint_divide(void *x, unsigned int uy, 
                ARBPRECTYPE in_type) {
            ArbPrecData ap;
            if (in_type == UINT32) {
                ap.uint32 = uint32_t(EXTRACT(uint32_t, x)) / uy;
            } else if (in_type == UINT32) {
                ap.int32 = int32_t(EXTRACT(int32_t, x)) / uy;
            } else if (in_type == FP32) {
                ap.fp32 = float(EXTRACT(float, x)) / uy;
            }
            return ap;
        }

        static bool _gt(void *x, void *y, ARBPRECTYPE in_type) {
            bool gt;
            if (in_type == UINT32) {
                gt = uint32_t(EXTRACT(uint32_t, x)) > 
                    uint32_t(EXTRACT(uint32_t, y));
            } else if (in_type == UINT32) {
                gt = int32_t(EXTRACT(int32_t, x)) > 
                    int32_t(EXTRACT(int32_t, y));
            } else if (in_type == FP32) {
                gt = float(EXTRACT(float, x)) >
                    float(EXTRACT(float, y));
            }
            return gt;
        }

        static void _dump(FILE *f, void *x, ARBPRECTYPE type) {
            switch (type) {
                case UINT8:
                    fprintf(f, "%d", EXTRACT(uint8_t, x));
                    break;
                case INT8:
                    fprintf(f, "%d", EXTRACT(int8_t, x));
                    break;
                case UINT16:
                    fprintf(f, "%d", EXTRACT(uint16_t, x));
                    break;
                case INT16:
                    fprintf(f, "%d", EXTRACT(int16_t, x));
                    break;
                case UINT32:
                    fprintf(f, "%d", EXTRACT(uint32_t, x));
                    break;
                case INT32:
                    fprintf(f, "%d", EXTRACT(int32_t, x));
                    break;
                case FP16:
                case FP32:
                    fprintf(f, "%f", EXTRACT(float, x));
                    break;
                default:
                    assert(0);
            }
        }
};


#endif // ARB_PREC_H
