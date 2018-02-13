#ifndef __AL_HAL_UDMA_M2M_H__
#define __AL_HAL_UDMA_M2M_H__
/*******************************************************************************
Copyright (C) 2018 Annapurna Labs Ltd.

This file may be licensed under the terms of the Annapurna Labs Commercial
License Agreement.

Alternatively, this file can be distributed under the terms of the GNU General
Public License V2 as published by the Free Software Foundation and can be
found at http://www.gnu.org/licenses/gpl-2.0.html

Alternatively, redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following conditions are
met:

    *     Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

    *     Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/

/* 
 This is the higher level UDMA API for Kaena.
 - DMA engines are always bi-directional; each engine has both m2s and s2m 
   queues
 - The only memory operation supported by this API is m2m.  m2s and s2m queue
   with the same queue index are bundled together to provide m2m
 - m2m supports scatter/gather
 - each queue can have VMPR (overriding high bits of src/dst address)
*/

#include "al_hal_udma.h"

// init the engine
extern int al_udma_m2m_init(struct al_udma *udma, struct al_udma_params *udma_params);

// sets up DMA descriptors for the memcopy to be triggered by SP
extern int al_udma_m2m_copy_prepare(struct al_udma *udma, uint32_t qid, void* src, void* dst);

// sets up DMA descriptors and performs the actual copy
extern int al_udma_m2m_copy(struct al_udma *udma, uint32_t qid, void* src, void* dst);

#endif

