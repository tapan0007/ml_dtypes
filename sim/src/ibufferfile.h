#ifndef _IBUFFERFILE_H
#define _IBUFFERFILE_H

#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

#if __linux__
#include <linux/version.h>
#if LINUX_VERSION_CODE > KERNEL_VERSION(2,6,22)
#define _MAP_POPULATE_AVAILABLE
#endif
#endif

#ifdef _MAP_POPULATE_AVAILABLE
#define MMAP_FLAGS (MAP_PRIVATE | MAP_POPULATE)
#else
#define MMAP_FLAGS MAP_PRIVATE
#endif

class IBufferFile : public UopFeedInterface {
    public:
        IBufferFile(char* fname) {
            struct stat st;
            fd = open(fname, O_RDONLY, 0);
            assert(fd > 0);

            stat(fname, &st);
            fsize = st.st_size;

            auto rc = mmap(NULL, fsize, PROT_READ, MMAP_FLAGS, fd, 0);
            assert(rc != nullptr);

            data_idx = data_start = reinterpret_cast<char*>(rc);
        }
        ~IBufferFile() { munmap(data_start, fsize); }
        bool         empty()   {return data_idx == data_start + fsize;}
        void*        front()   {return data_idx;}
        void         pop()     {data_idx += INSTRUCTION_NBYTES;}
    private:
        int    fd = -1;
        size_t  fsize = 0;
        char   *data_start = nullptr;
        char   *data_idx = nullptr;
};

#endif
