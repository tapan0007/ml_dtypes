#ifndef PATCH_ITERATOR_H
#define PATCH_ITERATOR_H

class PatchIterator {
    public:
        /* Argument that sets up empty patch iterator that tracks an 
         * ndim-rectangle */
        PatchIterator(size_t _ndims) : ndims(_ndims) {
            steps.resize(ndims);
            cnts.resize(ndims + 1); /* create larger for overflow */
            nums.resize(ndims + 1);
            cnts[ndims] = 1;        /* mark as overflowed, for EOP */
        }
        /* load the Patch Iterator with dimensions and steps */
        void init(int *dims, int *step) {
            for (size_t i = 0; i < ndims; i++) {
                nums[i] = dims[i];
                steps[i] = step[i];
            }
            cnts[ndims] = 0;
            nums[ndims] = std::numeric_limits<std::size_t>::max();
        }
        /* clear the count state */
        void reset() {
            for (size_t i = 0; i <= ndims; i++) {
                cnts[i] = 0;
            }
        }
        /* step the coordinates of the patch */
        void increment() {
            bool rollover = true;
            for (size_t i = 0; (i <= ndims) && rollover; i++) {
                cnts[i]++;
                rollover = (cnts[i] >= nums[i]);
                if (rollover) {
                    cnts[i] = 0;
                }
            }
        }
        /* get the current 1D coordinate */
        size_t coordinates() {
            size_t coord = 0;
            for (size_t i = 0; i < ndims; i++) {
                coord += cnts[i] * steps[i];
            }
            return coord;
        }
        /* end of patch - aka. did we overflow */
        bool eop() {
            return (cnts[ndims] > 0);
        }
        /* on last coordinate of patch */
        bool last() {
            bool last = true;
            for (size_t i = 0; i < ndims && last; i++) {
                last &= (cnts[i] == (nums[i] - 1));
            }
            last &= !cnts[ndims];
            return last;
        }
        /* are these coordinates in the range of the patch */
        bool in_range(int *r) {
            bool in_range = true;
            for (size_t i = 0; (i < ndims) && in_range; i++) {
                in_range &= (r[i] >= 0) && ((size_t )r[i] < nums[i]);
            }
            return in_range;
        }
        /* indexing, to figure out current coordinates, if we must */
        size_t& operator[](int index) { return cnts[index]; }
    private:
        std::vector<size_t> nums;
        std::vector<size_t> cnts;
        std::vector<int>    steps;
        size_t ndims = {0};
};

#endif
