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
                std::cout << "YELP YELP";
                std::cout << x.type;
                std::cout << y.type;
                std::cout << std::flush;

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
                std::cout << "HELP HELP";
                std::cout << x.type;
                std::cout << y.type;
                std::cout << std::flush;

                assert(0 && "unsupported combo");
            }
            return ap;
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
