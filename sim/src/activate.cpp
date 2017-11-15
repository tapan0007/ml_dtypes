#include "activate.h"
#include "io.h"
#include <math.h>
#include <algorithm>

//-----------------------------------------------------------------------
//  Activate
//-----------------------------------------------------------------------
void
Activate::connect(ActivateInterface *connection)
{
    this->connection = connection;
}

ActivateSignals
Activate::pull_activate()
{
    return as;
}

void
Activate::step()
{
    as = connection->pull_activate();
    if (as.valid) {
	ArbPrecData raw_pixel;
	ArbPrecData in_pixel;
	ArbPrecData out_pixel;
        ARBPRECTYPE in_dtype  = as.in_dtype;
        ARBPRECTYPE out_dtype = as.out_dtype;
	size_t in_dsize  = sizeofArbPrecType(in_dtype);
	size_t out_dsize = sizeofArbPrecType(out_dtype);
	src_partition_size = (as.src_addr.sys >= MMAP_PSUM_BASE) ?
            SZ(COLUMN_SIZE_BITS) : SZ(ROW_SIZE_BITS);
        dst_partition_size = (as.dst_addr.sys >= MMAP_PSUM_BASE) ?
            SZ(COLUMN_SIZE_BITS) : SZ(ROW_SIZE_BITS);
	memory->read_global(&raw_pixel, as.src_addr.sys, in_dsize);
        in_pixel = ArbPrec::cast_to_fp32(raw_pixel, in_dtype);
        float in_fp32 = in_pixel.fp32;

        switch (as.func) {
            case RELU:
                /* max(0, x) */
                if (in_fp32 > 0) {
                    out_pixel.fp32 = in_fp32;
                } 
                break;
            case LEAKY_RELU: 
                {
                    /* max(x, 0.1 * x) */
                    float point01 = 0.01;
                    out_pixel.fp32 = std::max(in_fp32, point01 * in_fp32);
                }
                break;
            case SIGMOID:
                /* performing 1/(1+math.exp(-x)) */
                out_pixel.fp32 = 1.0 / (1 + exp(-in_fp32));
                break;
            case TANH:
                /* tanh(x) */
                out_pixel.fp32 = std::tanh((double)in_fp32); 
                break;
            case IDENTITY:
                out_pixel = in_pixel;
                break;
            default:
                break;
	}
        out_pixel = ArbPrec::cast_from_fp32(out_pixel, out_dtype);
	memory->write_global(as.dst_addr.sys, &out_pixel, out_dsize);
	printf("Activate\n");
        ArbPrec::dump(stdout, out_pixel, out_dtype);
        printf("\n");
	as.src_addr.sys += src_partition_size;
	as.dst_addr.sys += dst_partition_size;
        as.valid = ((--as.countdown) > 0);
    }
}

//-----------------------------------------------------------------------
//  ActivateArray
//-----------------------------------------------------------------------
ActivateArray::ActivateArray(MemoryMap *mmap, size_t n_cols) {
    for (size_t i = 0; i < n_cols; i++) {
        activators.push_back(Activate(mmap));
    }
    for (size_t i = 1; i < activators.size(); i++) {
        activators[i].connect(&activators[i-1]);
    }
}

Activate& ActivateArray::operator[](int index){
    return activators[index];
}

void
ActivateArray::connect(ActivateInterface *ai)
{
    activators[0].connect(ai);
}

void
ActivateArray::step() {
    for (int i = activators.size() - 1; i >= 0; i--) {
        activators[i].step();
    }
}
