#ifndef SP_ISA_WRITE4B
#define SP_ISA_WRITE4B

#include "sp_isa.h"

struct  SP_CMD_WRITE_4B {
	struct SP_CMD_HEADER	hdr; 
	uint8_t	reserved[2];
	uint32_t	wdata;
	uint64_t	address;
	SP_CMD_WRITE_4B(uint64_t address,uint32_t value) : 
		address(address),wdata(value) {
		hdr.type= WRITE_4B;
		hdr.dword_len=4;
		tonga_assert(address);
		tonga_assert(sizeof(*this)==hdr.dword_len*4);
	};
} TONGA_PACKED;


#endif //SP_ISA_WRITE4B
