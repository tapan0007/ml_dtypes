#include "activate.h"
#include "io.h"

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
	ArbPrecData in_pixel ;
	ArbPrecData out_pixel;
        ARBPRECTYPE dtype = as.dtype;
	size_t dsize = sizeofArbPrecType(dtype);
	src_partition_size = (as.src_addr.sys >= MMAP_PSUM_BASE) ?
            SZ(COLUMN_SIZE_BITS) : SZ(ROW_SIZE_BITS);
        dst_partition_size = (as.dst_addr.sys >= MMAP_PSUM_BASE) ?
            SZ(COLUMN_SIZE_BITS) : SZ(ROW_SIZE_BITS);
	memory->read_global(&in_pixel, as.src_addr.sys, dsize);
        switch (as.func) {
            case RELU:
                {
                    /* max(0, x) */
                    ArbPrecData zero_pixel;
                    if (ArbPrec::gt(in_pixel, zero_pixel, dtype)) {
                        out_pixel = in_pixel;
                    }
                }
                break;
            case LEAKY_RELU:
                {
                    /* max(x, 0.01*x) */
                    ArbPrecData cmp_pixel = ArbPrec::uint_divide(in_pixel, 10,
                            dtype);
                    if (ArbPrec::gt(in_pixel, cmp_pixel, dtype)) {
                        out_pixel = in_pixel;
                    } else {
                        out_pixel = cmp_pixel;
                    }
                }
                break;
            case SIGMOID:
                /* performing 1/(1+math.exp(-x)) */
                assert(0 && "to be implemented");
                //out_pixel = ArbPrec::sigmoid(in_pixel, dtype);
                break;
            case TANH:
                assert(0 && "to be implemented");
                //out_pixel = ArbPrec::tanh(in_pixel, dtype);
                break;
            case IDENTITY:
                break;
            default:
                break;
	}
	memory->write_global(as.dst_addr.sys, &out_pixel, dsize);
	printf("Activate\n");
        ArbPrec::dump(stdout, out_pixel, dtype);
        printf("\n");
	as.src_addr.sys += src_partition_size;
	as.dst_addr.sys += dst_partition_size;
        as.valid = ((as.countdown--) > 0);
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
