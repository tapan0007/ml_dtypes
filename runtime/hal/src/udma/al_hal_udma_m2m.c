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

#include "al_hal_udma_m2m.h"

static int init_one_q(struct al_udma* udma, int qid)
{
    int ret;
    struct al_udma_q_params qp;
    // TODO same number of descriptors for each queue. 
    // In the future I might need to parameterize this depending
    // on use cases for each queue
#define NDESC (1<<12)
    // TODO what's my CDESC size ??
#define CDESC_SIZE 8
    memset(&qp, 0, sizeof(qp));
    qp.adapter_rev_id = 0;
    qp.type = UDMA_TX;
    qp.size = NDESC;
    qp.desc_base = 0; // TODO alloc
    qp.desc_phy_base = 0; // TODO alloc
    qp.cdesc_base = 0; // TODO need for TX?
    qp.cdesc_phy_base = 0; // TODO
    qp.cdesc_size = CDESC_SIZE;
    ret = al_udma_q_init(udma, qid, &qp);
    if( ret ) {
        al_err("udma_m2m: failed to init m2s queue %d, err: %d\n", qid, ret);
        return ret;
    }
    memset(&qp, 0, sizeof(qp));
    qp.adapter_rev_id = 0;
    qp.type = UDMA_RX;
    qp.size = NDESC;
    qp.desc_base = 0; // TODO
    qp.desc_phy_base = 0; // TODO
    qp.cdesc_base = 0; // TODO
    qp.cdesc_phy_base = 0; // TODO
    qp.cdesc_size = CDESC_SIZE;
    ret = al_udma_q_init(udma, qid, &qp);
    if( ret ) {
        al_err("udma_m2m: failed to init m2s queue %d, err: %d\n", qid, ret);
        return ret;
    }
    return ret;
}

int al_udma_m2m_init(struct al_udma *udma, void __iomem *regs_base)
{
    int ret, i;
    struct al_udma_params params;
    al_assert(udma);
    al_assert(regs_base);
    params.udma_regs_base = regs_base;
    params.num_of_queues = DMA_MAX_Q_V4;
    params.name = "Tonga UDMA";
    ret = al_udma_init(udma, &params);
    if( ret ) {
        al_err("udma_m2m: failed to init engine: %p, err: %d\n", regs_base, ret);
        return ret;
    }
    // init all the queues, the queues are bi-directional
    // so need to init both Tx and Rx 
    for( i = 0; i < params.num_of_queues; i++ ) {
        ret = init_one_q(udma, i);
        if( ret ) {
            al_err("udma_m2m: failed to init queue: %d, err: %d\n", i, ret);
            break;
        }
    }
    return al_udma_state_set(udma, UDMA_NORMAL);
}

#define __unused __attribute__((unused))

// sets up DMA descriptors for the memcopy to be triggered by SP
static int al_udma_m2m_copy_prepare_impl(struct al_udma_q* txq, struct al_udma_q* rxq, struct al_udma_sgl* src, struct al_udma_sgl* dst)
{
    union al_udma_desc* desc;
    uint32_t ndesc;
    uint32_t flags_len;
    uint32_t i, last_sg;
#ifdef DEBUG
    {
        uint32_t size1 = 0, size2 = 0;
        for( i = 0; i < src->num_sg; i++ ) 
            size1 += src->sg[i].len;
        for( i = 0; i < dst->num_sg; i++ ) 
            size2 += dst->sg[i].len;
        al_assert(size1 == size2);
    }
#endif
    al_assert(txq);
    al_assert(rxq);
    // enough room?
    ndesc = al_udma_available_get(txq);
    if( ndesc < src->num_sg ) {
        al_err("udma_m2m: not enough room in TX queue %d, requested: %u, available: %u\n", txq->qid, src->num_sg, ndesc);
        return -ENOMEM;
    }
    ndesc = al_udma_available_get(rxq);
    if( ndesc < dst->num_sg ) {
        al_err("udma_m2m: not enough room in RX queue %d, requested: %u, available: %u\n", rxq->qid, dst->num_sg, ndesc);
        return -ENOMEM;
    }
    // setup TX (m2s)
    last_sg = src->num_sg - 1;
    flags_len = 0;
    for( i = 0; i <= last_sg; i++ ) {
        // TODO does it make sense to add a function that gets >1
        desc = al_udma_desc_get(txq);
        flags_len |= al_udma_ring_id_get(txq) << AL_M2S_DESC_RING_ID_SHIFT;
        if( i == 0 ) 
            flags_len |= AL_M2S_DESC_FIRST;
        if( i == last_sg ) 
            flags_len |= AL_M2S_DESC_LAST;
        flags_len |= src->sg[i].len & AL_M2S_DESC_LEN_MASK;
        desc->tx.len_ctrl = swap32_to_le(flags_len);
        desc->tx.buf_ptr = swap64_to_le(src->sg[i].addr);
        desc->tx.meta_ctrl = 0;
    }
    // setup RX (s2m)
    flags_len = 0;
    for( i = 0; i < dst->num_sg; i++ ) {
        // TODO does it make sense to add a function that gets >1
        desc = al_udma_desc_get(rxq);
        flags_len |= al_udma_ring_id_get(rxq) << AL_M2S_DESC_RING_ID_SHIFT;
        flags_len |= dst->sg[i].len & AL_M2S_DESC_LEN_MASK;
        desc->rx.len_ctrl = swap32_to_le(flags_len);
        desc->rx.buf1_ptr = swap64_to_le(dst->sg[i].addr);
    }
    return 0;
}

int al_udma_m2m_copy_prepare(struct al_udma *udma, uint32_t qid, struct al_udma_sgl* src, struct al_udma_sgl* dst)
{
    struct al_udma_q* txq;
    struct al_udma_q* rxq;
    int ret;

    al_assert(udma);
    al_assert(qid < udma->num_of_queues);
    ret = al_udma_q_handle_get(udma, qid, UDMA_TX, &txq);
    if( ret ) {
        al_err("udma_m2m: failed to get TX queue: %d, err: %d\n", qid, ret);
        return ret;
    }
    ret = al_udma_q_handle_get(udma, qid, UDMA_TX, &rxq);
    if( ret ) {
        al_err("udma_m2m: failed to get TX queue: %d, err: %d\n", qid, ret);
        return ret;
    }
    return al_udma_m2m_copy_prepare_impl(txq, rxq, src, dst);
}

// sets up DMA descriptors and performs the actual copy
int al_udma_m2m_copy(struct al_udma *udma, uint32_t qid, struct al_udma_sgl* src, struct al_udma_sgl* dst)
{
    struct al_udma_q* txq;
    struct al_udma_q* rxq;
    int ret;

    al_assert(udma);
    al_assert(qid < udma->num_of_queues);
    ret = al_udma_q_handle_get(udma, qid, UDMA_TX, &txq);
    if( ret ) {
        al_err("udma_m2m: failed to get TX queue: %d, err: %d\n", qid, ret);
        return ret;
    }
    ret = al_udma_q_handle_get(udma, qid, UDMA_TX, &rxq);
    if( ret ) {
        al_err("udma_m2m: failed to get TX queue: %d, err: %d\n", qid, ret);
        return ret;
    }
    ret = al_udma_m2m_copy_prepare_impl(txq, rxq, src, dst);
    if( ret == 0 ) {
        al_udma_desc_action_add(rxq, dst->num_sg);
        al_udma_desc_action_add(txq, src->num_sg);
    }
    return ret;
}

