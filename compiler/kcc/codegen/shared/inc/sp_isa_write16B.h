#ifndef SP_ISA_WRITE16B
#define SP_ISA_WRITE16B

#include "sp_isa.hh"

struct  SP_CMD_WRITE_16B {
	struct SP_CMD_HEADER	hdr; 
	uint8_t	reserved[2];
	uint32_t	reservedd;
	uint64_t	address;
	uint32_t	wdata[4];
	SP_CMD_WRITE_16B(uint64_t address,void* data) : 
		eddress(address) {
		hdr.type= WRITE_16B;
		hdr.dword_len=8;
		tonga_assert(data);
		wdata[0]=*((uint32_t*)data);
		wdata[1]=*((uint32_t*)data + 1);
		wdata[2]=*((uint32_t*)data + 2);
		wdata[3]=*((uint32_t*)data + 3);

		tonga_assert(address);
		tonga_assert(sizeof(*this)==hdr.dword_len*4);
	};
} TONGA_PACKED;



#endif //SP_ISA_WRITE16B
