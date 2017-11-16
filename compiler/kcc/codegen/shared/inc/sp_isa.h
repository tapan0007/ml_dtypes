#ifndef SP_ISA_H
#define SP_ISA_H

#include <stdio.h>
#include <stdint.h>
#include "isa_common.h"
#include "sp_isa_notify.h"
#include "sp_isa_pause.h"
#include "sp_isa_waitevent.h"
#include "sp_isa_waittime.h"
#include "sp_isa_write16B.h"
#include "sp_isa_write4B.h"

enum SP_CMD_TYPE {
    UNDEF = 0,
    WRITE_4B=1,
    WRITE_16B=2,
    NOTIFY=8,
    WAIT_TIME=9,
    WAIT_EVENT=0xA,
    PAUSE_SP=0xF,
    SP_CMD_TYPE_MAX=0x10
};


struct SP_CMD_HEADER {
    uint8_t         phase : 1;     
    uint8_t         type  : 7;       
    uint8_t         dword_len : 8;   
    void            set_phase(uint8_t ph) { 
        phase=ph & 0x1; 
    }; 
} TONGA_PACKED;




#endif
