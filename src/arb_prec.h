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

class ArbPrec {
    public:
        ArbPrec() : type(INVALID_ARBPRECTYPE) {}
        ArbPrec(uint8_t _val) : uint8(_val), type(UINT8) {}
        ArbPrec(uint32_t _val) : uint32(_val), type(UINT32) {}
        ArbPrec(float _val) : fp32(_val), type(FP32) {}
        ArbPrec(ARBPRECTYPE _type) : uint8(0), uint32(0), fp32(0), type(_type) {};
        ArbPrec(ARBPRECTYPE _type, int val) {
            switch (_type) {
                case UINT8:
                    ArbPrec((uint8_t)val); break;
                case UINT32:
                    ArbPrec((uint32_t)val); break;
                case FP32:
                    ArbPrec((float)val); break;
                default:
                    assert(0);
            }
        }
        ~ArbPrec() {};
        friend ArbPrec operator*(const ArbPrec &x, const ArbPrec &y) {
            ArbPrec ap;
            if (x.type == UINT8 and y.type == UINT8) {
                ap = ArbPrec(uint32_t(x.uint8) * uint32_t(y.uint8));
            } else if (x.type == UINT8 and y.type == FP32) {
                ap = ArbPrec(x.uint8 * y.fp32);
            } else if (x.type == FP32 and y.type == UINT8) {
                ap = ArbPrec(x.fp32 * y.uint8);
            } else {
                assert(0 && "unsupported combo");
            }
            return ap;
        }

        friend ArbPrec operator/(const ArbPrec &x, const ArbPrec &y) {
            ArbPrec ap;
            if (x.type == UINT32 and y.type == UINT32) {
                ap = ArbPrec(x.uint32 / y.uint32);
            } else if (x.type == FP32 and y.type == FP32) {
                ap = ArbPrec(x.fp32 / y.fp32);
            } else {
                assert(0 && "unsupported combo");
            }
            return ap;
        }

        friend ArbPrec operator+(const ArbPrec &x, const ArbPrec &y) {
            ArbPrec ap;
            if (x.type == UINT32 and y.type == UINT32) {
                ap = ArbPrec(x.uint32 + y.uint32);
            } else if (x.type == FP32 and y.type == FP32) {
                ap = ArbPrec(x.fp32 + y.fp32);
            } else {
                assert(0 && "unsupported combo");
            }
            return ap;
        }
        friend bool operator>(const ArbPrec &x, const ArbPrec &y) {
            if (x.type == UINT32 and y.type == UINT32) {
                return x.uint32 > y.uint32;
            } else if (x.type == FP32 and y.type == FP32) {
                return x.fp32 > y.fp32;
            } 
            assert(0);
        }
        void *raw_ptr() {
            static void *type_to_ptr[NUM_ARBPRECTYPE] = {[UINT8]=&this->uint8, [UINT32]=&this->uint32, [FP32]=&this->fp32};
            return type_to_ptr[type];
        }
        int nbytes() {
            static int type_to_nbytes[NUM_ARBPRECTYPE] = {[UINT8]=sizeof(uint8_t), [UINT32]=sizeof(uint32_t), [FP32]=sizeof(float)};
            return type_to_nbytes[type];
        }

        void dump(FILE *f) {
            switch (type) {
                case UINT8:
                    fprintf(f, "%d", uint8);
                    break;
                case UINT32:
                    fprintf(f, "%d", uint32);
                    break;
                case FP32: 
                    fprintf(f, "%f", fp32);
                    break;
                default:
                    assert(0);
            }
        }
        uint8_t uint8;
        uint32_t uint32;
        float   fp32;
    private:
        ARBPRECTYPE type;
};

#endif // ARB_PREC_H
