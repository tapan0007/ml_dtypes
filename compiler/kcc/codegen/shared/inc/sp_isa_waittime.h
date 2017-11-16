#ifndef SP_ISA_WAITTIME
#define SP_ISA_WAITTIME

#include "spa_isa.h"


struct  SP_CMD_WAIT_TIME {
	struct SP_CMD_HEADER	hdr; 
	uint8_t		reserved[2];
	uint32_t	delay_in_clk;
	SP_CMD_WAIT_TIME(uint32_t delay) : 
		delay_in_clk(delay){
		hdr.type=WAIT_TIME;
		hdr.dword_len=2;
		tonga_assert(sizeof(*this)==hdr.dword_len*4);
	};
} TONGA_PACKED;


#endif //SP_ISA_WAITTIME
