#ifndef SP_ISA_WAITEVENT_H
#define SP_ISA_WAITEVENT_H

#include "spa_isa.h"
#include "events.h"

struct  SP_CMD_WAIT_EVENT {
	struct SP_CMD_HEADER	hdr; 
	uint8_t		reserved[2];
	uint8_t		event_id;
	uint8_t		reservedd[3];
	SP_CMD_WAIT_EVENT(AWSEVENTS::Event* event) : 
		event_id(event->get_local_id()){
		hdr.type=WAIT_EVENT;
		hdr.dword_len=2;
		tonga_assert(sizeof(*this)==hdr.dword_len*4);
	};
} TONGA_PACKED;


#endif //SP_ISA_WAITEVENT_H
