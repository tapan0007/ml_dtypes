#ifndef _UTILS_H
#define _UTILS_H
#include "tpb_isa.h"


void
get_dims(char *fname, uint64_t *shape, unsigned int &word_size, 
	unsigned int &tot_size, ARBPRECTYPE &dtype) {
    bool fortran_order;
    unsigned int ndims;
    char ctype;
    FILE *fp = fopen(fname, "r");
    unsigned int *ui_shape;
    cnpy::parse_npy_header(fp, word_size, ui_shape, ndims, fortran_order, ctype);
    tot_size = word_size;

    assert(ndims = 4);
    for (size_t i = 0; i < ndims; i++) {
       shape[i] = ui_shape[i];
       tot_size *= ui_shape[i];
    }

    switch (word_size) {
	case 1:
	    switch (ctype) {
		case 'u':
		    dtype = UINT8;
		    break;
		case 'i':
		    dtype = INT8;
		    break;
		default:
		    assert(0);
	    }
	    break;
	case 2:
	    switch (ctype) {
		case 'u':
		    dtype = UINT8;
                    break;
                case 'i':
                    dtype = INT8;
                    break;
                case 'f':
                    dtype = FP16;
                    break;
                default:
                    assert(0);
            }
            break;
        default:
            assert(0);
    }
}


#endif
