#include "pe.h"
#include "pe_array.h"
#include "state_buffer.h"
#include <array>
#include <cstdint>

ProcessingElementArray::ProcessingElementArray(int _n_rows, int _n_cols) : n_rows(_n_rows), n_cols(_n_cols)  {
    // create 2d array
    pe_array = new ProcessingElement*[n_rows];
    for (int i = 0; i < n_rows; i++) {
        pe_array[i] = new ProcessingElement[n_cols];
    }

    // connect 2d array together
    for (int i = n_rows - 1; i >= 0; i--) {
        for (int j = n_cols - 1; j >= 0; j--) {
            if ((i -1 >= 0)) {
                pe_array[i][j].connect_north(&pe_array[i-1][j]);
            } else {
                // zeros always come from ns
                pe_array[i][j].connect_north(&ns_generator);
            }
            if ((j -1 >= 0)) {
                pe_array[i][j].connect_west(&pe_array[i][j-1]);
            }
        }
    }
}

ProcessingElementArray::~ProcessingElementArray() {
    for (int i = 0; i < n_rows; i++) {
        delete [] pe_array[i];
    }
    delete [] pe_array;
}

ProcessingElement*&
ProcessingElementArray::operator[](int index) {
    return pe_array[index];
}

int
ProcessingElementArray::num_rows() {
    return n_rows;
}

int
ProcessingElementArray::num_cols() {
    return n_cols;
}

void
ProcessingElementArray::connect_west(int row, PeEWInterface *ew) {
    pe_array[row][0].connect_west(ew);
}

void
ProcessingElementArray::connect_north(int col, PeNSInterface *ns) {
    pe_array[0][col].connect_north(ns);
}

void
ProcessingElementArray::connect_statebuffer(int row, SbEWBroadcastInterface *sb) {
    for (int j = 0; j < n_cols; j++) {
        pe_array[row][j].connect_statebuffer(sb);
    }
}

void
ProcessingElementArray::step() {
    for (int i = n_rows - 1; i >= 0; i--) {
        for (int j = n_cols - 1; j >= 0; j--) {
            pe_array[i][j].step();
        }
    }
}

void ProcessingElementArray::dump(FILE *f) {
    for (int i = 0; i < n_rows; i++) {
        fprintf(f, "row=%d ", i);
        for (int j = 0; j < n_cols; j++) {
            pe_array[i][j].dump(f);
        }
        fprintf(f, "\n");
    }
}
