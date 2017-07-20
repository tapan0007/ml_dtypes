#include "pe.h"
#include "pe_array.h"
#include "state_buffer.h"
#include <array>
#include <cstdint>

ProcessingElementArray::ProcessingElementArray(int _num_rows, int _num_cols) : num_rows(_num_rows), num_cols(_num_cols)  {
    // create 2d array
    pe_array = new ProcessingElement*[num_rows];
    for (int i = 0; i < num_rows; i++) {
        pe_array[i] = new ProcessingElement[num_cols];
    }

    // connect 2d array together
    NSInterface *north;
    EWInterface *west;
    RandomInterfaces *ri = new RandomInterfaces();
    for (int i = num_rows - 1; i >= 0; i--) {
        for (int j = num_cols - 1; j >= 0; j--) {
            if ((i -1 >= 0)) {
                north = &pe_array[i-1][j];
            } else {
                north = ri;
            }
            if ((j -1 >= 0)) {
                west =  &pe_array[i][j-1];
            } else {
                west = ri;
            }
            pe_array[i][j].connect(west, north);
        }
    }

}

ProcessingElementArray::~ProcessingElementArray() {
    delete [] pe_array;
}

void
ProcessingElementArray::step() {
    for (int i = num_rows - 1; i >= 0; i--) {
        for (int j = num_cols - 1; j >= 0; j--) {
            pe_array[i][j].step();
        }
    }
}

void ProcessingElementArray::dump(FILE *f) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            pe_array[i][j].dump(f);
        }
        fprintf(f, "\n");
    }
}
