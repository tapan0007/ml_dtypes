#ifndef SP_ISA_NOTIFY_H
#define SP_ISA_NOTIFY_H

#include "sp_isa.h"

struct  SP_CMD_NOTIFY {
	struct SP_CMD_HEADER	hdr; 
	uint8_t		notifQ_id;
	uint8_t		reserved;
	uint32_t	token;
	SP_CMD_NOTIFY(uint8_t qid,uint32_t tok) : 
		notifQ_id(qid), token(tok) {
		hdr.type=NOTIFY;
		hdr.dword_len=2;
//		tonga_assert(qid<MAX_QID_PER_SP);
		tonga_assert(sizeof(*this)==hdr.dword_len*4);
	};
} TONGA_PACKED;


#endif //SP_ISA_NOTIFY_H

