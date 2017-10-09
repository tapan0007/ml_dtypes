#ifndef SP_ISA_PAUSE_H
#define SP_ISA_PAUSE_H

#include "sp_isa.h"

struct  SP_CMD_PAUSE {
	struct SP_CMD_HEADER	hdr; 
	uint8_t		reserved[2];
	uint32_t	reservedd;
	SP_CMD_PAUSE() { 
		hdr.type=PAUSE_SP;
		hdr.dword_len=2;
		tonga_assert(sizeof(*this)==hdr.dword_len*4);
	};
} TONGA_PACKED;


#endif //SP_ISA_PAUSE_H
