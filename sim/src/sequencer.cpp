#include <assert.h>
#include "sequencer.h"
#include "uarch_defines.h"
#include "string.h"
#include "tpb_isa.h"
#include "io.h"

extern class Memory memory;
#define UNUSED(X) (void)(X)

/*------------------------------------
 * EdgeSignalsInstruction
 *------------------------------------ */
template<>
void
DynamicInstruction<EdgeSignals>::execute(void *v_seq) {
    Sequencer *seq = (Sequencer *)v_seq;


    seq->es = args;
    seq->raw_signal = true;
}

/*------------------------------------
 * RdIfmap
 *------------------------------------ */
template<>
void  DynamicInstruction<SIM_RDIFMAP>::execute(void *v_seq) {
    UNUSED(v_seq);
    void *i_ptr;
    int i_n, i_c, i_h, i_w;
    size_t word_size;

    /* load io_mmap */
    i_ptr = memory.io_mmap(args.fname, i_n, i_c, i_h, i_w, word_size);
    memory.bank_mmap(args.address, i_ptr, i_c, i_h * i_w * word_size);
}

/*------------------------------------
 * RdFilterArgs
 *------------------------------------ */
template<>
void  DynamicInstruction<SIM_RDFILTER>::execute(void *v_seq) {
    UNUSED(v_seq);
    void *f_ptr;
    int r,s,t,u;
    int w_c, w_m, w_r, w_s;
    size_t word_size;

    /* load io_mmap */
    f_ptr = memory.io_mmap(args.fname, r, s, t, u, word_size);
    memory.swap_axes(f_ptr, r, s, t, u, word_size);
    w_c = s; // for swamp, M now corresponds to C
    w_m = r; // for swamp, C now corresponds to M
    w_r = t;
    w_s = u;
    memory.bank_mmap(args.address, f_ptr, w_c, w_m * w_r * w_s * word_size);
}

/*------------------------------------
 * WrOfmapArgs
 *------------------------------------ */
template<>
void  DynamicInstruction<SIM_WROFMAP>::execute(void *v_seq) {
    UNUSED(v_seq);
    void *o_ptr;
    
    o_ptr = memory.sbuffer_bank_munmap(args.address, args.dims[1], 
            args.dims[2] * args.dims[3] * args.word_size);
    memory.io_write(args.fname, o_ptr, args.dims[0], args.dims[1], 
            args.dims[2], args.dims[3], args.word_size);
}


/*------------------------------------
 * LdWeightsInstruction
 *------------------------------------ */
template<>
void  DynamicInstruction<LDWEIGHTS>::execute(void *v_seq) {
    Sequencer *seq = (Sequencer *)v_seq;
    uint64_t num_cols = args.x_num * args.y_num; 
    assert(num_cols <= Constants::columns);
    seq->es.weight_valid = true;
    seq->es.weight_dtype = (ARBPRECTYPE) args.dtype;
    //seq->es.weight_full_addr = args.address;
    seq->weight_base = args.address;
    seq->es.weight_clamp = (num_cols == 1);
    seq->es.row_valid = true;
    seq->es.row_countdown = args.last_row; 
    seq->weight_clamp_countdown = num_cols;
    seq->weight_x_step = args.x_step;
    seq->weight_y_step = args.y_step;
    seq->weight_x_num = args.x_num;
    seq->weight_y_num = args.y_num;
    seq->weight_x_cnt = 0;
    seq->weight_y_cnt = 0;
    seq->raw_signal = false;
    seq->es.weight_full_addr = args.address + (args.y_num * args.y_step - 1) * 
        sizeofArbPrecType((ARBPRECTYPE)args.dtype);

}

/*------------------------------------
 * Convolve
 *------------------------------------ */
template<>
void  DynamicInstruction<MATMUL>::execute(void *v_seq) {
    Sequencer *seq = (Sequencer *)v_seq;
    /* ifmap setup */
    seq->es.ifmap_full_addr    = args.fmap_start_addr;
    seq->es.ifmap_dtype = (ARBPRECTYPE) args.dtype;
    seq->es.row_countdown = args.last_row; 
    seq->es.row_valid = true;
    seq->es.weight_toggle = args.toggle_weight;
    seq->ifmap_base = args.fmap_start_addr;
    seq->ifmap_x_num = args.fmap_x_num;
    seq->ifmap_y_num = args.fmap_y_num;
    seq->ifmap_x_cnt = 0;
    seq->ifmap_y_cnt = 0;
    seq->ifmap_x_step = args.fmap_x_step;
    seq->ifmap_y_step = args.fmap_y_step;
    seq->raw_signal = false;

    /* psum setup */
    seq->es.column_countdown = args.last_col;
    seq->es.column_valid = true;
    seq->es.psum_start = args.start_tensor_calc;
    seq->es.psum_stop = args.stop_tensor_calc;
    seq->es.psum_full_addr = args.psum_start_addr;

    /* signals that will stay constant for entire convolution */
    /* padding setup */
    seq->pad[N] = args.n_pad;
    seq->pad[S] = args.s_pad;
    seq->pad[E] = args.e_pad;
    seq->pad[W] = args.w_pad;
    /* pifmap is padded ifmap */
    seq->pifmap_x_cnt = 0;
    seq->pifmap_y_cnt = 0;
    seq->pifmap_x_num = seq->ifmap_x_num + args.e_pad + args.w_pad;
    seq->pifmap_y_num = seq->ifmap_y_num + args.n_pad + args.s_pad;
    seq->es.pad_valid = seq->pad_valid(0, 0);
    seq->es.ifmap_valid   = !seq->es.pad_valid;

}

/*------------------------------------
 * Pool
 *------------------------------------ */
template<>
void  DynamicInstruction<POOL>::execute(void *v_seq) {
    Sequencer *seq = (Sequencer *)v_seq;
    POOLFUNC pool_func = (POOLFUNC)args.pool_func;
	seq->ps.valid = true;
    seq->ps.func = pool_func;
	seq->ps.dtype = (ARBPRECTYPE)args.in_dtype;
	seq->ps.src_full_addr = args.src_start_addr;
	seq->ps.start = true;
	seq->ps.stop = (pool_func == IDENTITY_POOL) ||
        (args.src_x_num + args.src_y_num == 2);
	seq->ps.dst_full_addr = args.dst_start_addr;
	seq->ps.countdown = args.max_partition;
	
	seq->pool_src_base = args.src_start_addr;
	seq->pool_dst_base = args.dst_start_addr;
	seq->pool_src_x_cnt = 0;
	seq->pool_src_y_cnt = 0;
	seq->pool_src_x_step = args.src_x_step;
	seq->pool_src_y_step = args.src_y_step;
	seq->pool_src_x_num = args.src_x_num;
	seq->pool_src_y_num = args.src_y_num;

	seq->pool_str_x_cnt = 0;
	seq->pool_str_y_cnt = 0;
	seq->pool_str_x_step = args.str_x_step;
	seq->pool_str_y_step = args.str_y_step;
	seq->pool_str_x_num = args.str_x_num;
	seq->pool_str_y_num = args.str_y_num;


	seq->pool_dst_x_cnt = 0;
	seq->pool_dst_y_cnt = 0;
	seq->pool_dst_x_step = args.dst_x_step;
	seq->pool_dst_y_step = args.dst_y_step;
	seq->pool_dst_x_num = args.dst_x_num;
	seq->pool_dst_y_num = args.dst_y_num;

    seq->pool_eopools = false;
    /* emit one result for every stride... if not identity pool */
    assert((pool_func == IDENTITY_POOL) ||
            (seq->pool_dst_x_num * seq->pool_dst_y_num ==
             seq->pool_str_x_num * seq->pool_str_y_num));
}

/*------------------------------------
 * Sequencer
 *------------------------------------ */
void
Sequencer::connect_uopfeed(UopFeedInterface *_feed) {
    feed = _feed;
}

bool
Sequencer::synch() {
    static int pe_countdown = 128;
    bool busy = es.pad_valid || es.ifmap_valid  || es.weight_valid || 
        ps.valid;
    if (es.ifmap_valid) {
        pe_countdown = 128;
    } else if (ps.valid) {
        pe_countdown = 128;
    } else {
        if (pe_countdown) {
            pe_countdown--;
            busy = true;
        } 
    }
    return busy;
}


/* tells if you element (r,c) is a pad (!ifmap) element */
bool
Sequencer::pad_valid(uint8_t r, uint8_t c) {
    /* checks range, using outer ring of padding */
    return r < pad[N] || 
        ((r > (ifmap_y_num + pad[N] - 1)) && 
         (r < ifmap_y_num + pad[N] + pad[S])) ||
        c < pad[W] || 
        ((c > (ifmap_x_num + pad[W] - 1) && 
          (c < ifmap_x_num + pad[W] + pad[E])));
}


void
Sequencer::increment_and_rollover(uint8_t &cnt, uint8_t num, 
        uint8_t &rollover) {
    cnt++;
    if (cnt >= num) {
        cnt = 0;
        rollover++;
    }
}

#define COND_SET(X, VAL) (X == VAL) ? false : X=VAL

/* sub function of step - to step the edgesignal */
void
Sequencer::step_edgesignal() {
    /* UPDATE SEQUENCER STATE */
    if (es.pad_valid) {
        increment_and_rollover(pifmap_x_cnt, pifmap_x_num, pifmap_y_cnt);
    } 
    if (es.ifmap_valid) {
        increment_and_rollover(ifmap_x_cnt, ifmap_x_num, ifmap_y_cnt);
        increment_and_rollover(pifmap_x_cnt, pifmap_x_num, pifmap_y_cnt);
    }
    if (es.weight_valid) {
        increment_and_rollover(weight_x_cnt, weight_x_num, weight_y_cnt);
    }

    /* UPDATE SIGNALS */
    /* IFMAP/PAD */
    /* is the pad/ifmap valid or are we done? */
    es.pad_valid = pad_valid(pifmap_y_cnt, pifmap_x_cnt);
    if (!es.pad_valid && 
            (pifmap_y_cnt == pifmap_y_num)) { 
        /* clear state */
        es.ifmap_valid = false;
        es.psum_start = false;
        es.psum_stop = false;
        es.activation_valid = false;
    } else {
        es.ifmap_valid = !es.pad_valid;
    }

    /* figured out pad/ifmap valid, now compute addresses */
    if (es.ifmap_valid) {
        es.ifmap_full_addr = ifmap_base + 
            (ifmap_y_cnt * ifmap_y_step + ifmap_x_cnt * ifmap_x_step) * 
            sizeofArbPrecType((ARBPRECTYPE)es.ifmap_dtype);
    }
    /* Sending an ifmap, must be getting out ofmap! */
    if (es.ifmap_valid || es.pad_valid) {
        /* FIXME - this is the psum granularity, FIX */
        es.psum_full_addr += sizeofArbPrecType((ARBPRECTYPE)get_upcast(es.ifmap_dtype));
    }

    /*  WEIGHT */
    /* always toggle down weight */
    COND_SET(es.weight_toggle, false); 
    if (es.weight_valid) {
        if ((weight_x_cnt == weight_x_num - 1) &&
                (weight_y_cnt == weight_y_num -1)) {
            es.weight_clamp = true;
        } else if (weight_y_cnt ==  weight_y_num) {
            assert(es.weight_clamp);
            es.weight_clamp = false;
            es.weight_valid = false;
        } 
        if (es.weight_valid) {
            es.weight_full_addr = weight_base + 
                (weight_y_num * weight_y_step -
                 weight_y_cnt * weight_y_step -
                 weight_x_cnt * weight_x_step - 1) * 
                sizeofArbPrecType((ARBPRECTYPE)es.weight_dtype);
        }
    }
}

#define ROLLOVER(CNT, NUM, ROLLOVER) \
    if (CNT >= NUM) { \
        CNT = 0; \
        ROLLOVER++; \
    }


/* sub function of step - to step the poolsignal */
void
Sequencer::step_poolsignal() {
    uint32_t eopool = 0;
    bool     last_pool = false;
    bool     eostrides = false;
    if (pool_eopools) {
        ps.valid = false;
    }
    if (!ps.valid) {
        return;
    }
    size_t dsize = sizeofArbPrecType((ARBPRECTYPE)ps.dtype);

    pool_src_x_cnt++;
    ROLLOVER(pool_src_x_cnt, pool_src_x_num, pool_src_y_cnt);
    if (!pool_src_x_cnt) {
        ROLLOVER(pool_src_y_cnt, pool_src_y_num, eopool);
    }

    last_pool = 
        (pool_src_x_cnt == pool_src_x_num - 1) &&
        (pool_src_y_cnt == pool_src_y_num - 1);

    /* roll over dst counters */
    if (ps.func == IDENTITY_POOL || eopool) {
        pool_dst_x_cnt++;
        ROLLOVER(pool_dst_x_cnt, pool_dst_x_num, pool_dst_y_cnt);
        if (!pool_dst_x_cnt) {
            ROLLOVER(pool_dst_y_cnt, pool_dst_y_num, pool_eopools);
        }
        pool_str_x_cnt++;
        ROLLOVER(pool_str_x_cnt, pool_str_x_num, pool_str_y_cnt);
        if (!pool_str_x_cnt) {
            ROLLOVER(pool_str_y_cnt, pool_str_y_num, eostrides);
        }
    }
    assert((ps.func == IDENTITY_POOL) || (eostrides == pool_eopools));

    /* stopped last cycle, start anew */
    if (ps.func == IDENTITY_POOL) {
        ps.start = true;
        ps.stop = true;
    } else {
        ps.start = ps.stop; /* stopped last cycle, start anew */
        ps.stop = last_pool;
    }



    /* calculate address based on settings */
    if (ps.valid) {
        ps.src_full_addr = pool_src_base + 
            dsize * 
            (pool_str_x_cnt * pool_str_x_step +
             pool_str_y_cnt * pool_str_y_step) + /* tile */
            dsize *
            (pool_src_x_cnt * pool_src_x_step + /* offset in tile */
             pool_src_y_cnt * pool_src_y_step);
    }
    if (ps.start) {
        ps.dst_full_addr = pool_dst_base + 
            (pool_dst_x_cnt * pool_dst_x_step + 
             pool_dst_y_cnt * pool_dst_y_step) * dsize;
    }
}

void
Sequencer::step() {
    /* empty the instruction queue */
    Instruction *inst = NULL;
    if (!synch()) {
        if (!feed->empty()) {
            uint64_t *raw_inst = (uint64_t *)feed->front();
            switch (TPB_OPCODE(*raw_inst)) {
                case SIM_RDIFMAP_OPC:
                    inst = new DynamicInstruction<SIM_RDIFMAP>(
                            *((SIM_RDIFMAP *) (raw_inst)));
                    break;
                case SIM_RDFILTER_OPC:
                    inst = new DynamicInstruction<SIM_RDFILTER>(
                            *((SIM_RDFILTER *) (raw_inst)));
                    break;
                case SIM_WROFMAP_OPC:
                    inst = new DynamicInstruction<SIM_WROFMAP>(
                            *((SIM_WROFMAP *) (raw_inst)));
                    break;
                case LDWEIGHTS_OPC:
                    inst = new DynamicInstruction<LDWEIGHTS>(
                            *((LDWEIGHTS *) (raw_inst)));
                    break;
                case MATMUL_OPC:
                    inst = new DynamicInstruction<MATMUL>(
                            *((MATMUL *) (raw_inst)));
                    break;
                case POOL_OPC:
                    inst = new DynamicInstruction<POOL>(
                            *((POOL *) (raw_inst)));
                    break;
                default:
                    assert(0);
            }
            inst->execute(this);
            feed->pop();
        } else {
            es = {0};
        }
        return;
    }
    /* was the instruction a raw signal, if so, leave es alone and exit */
    if (raw_signal) {
        return;
    }

    step_edgesignal();
    step_poolsignal();

    clock++;
}


EdgeSignals Sequencer::pull_edge() {
    assert(!es.ifmap_valid ||
            ((es.ifmap_full_addr >= MMAP_SB_BASE) && 
             (es.ifmap_full_addr < ROW_SIZE)));
    assert(!es.weight_valid ||
          ((es.weight_full_addr >= MMAP_SB_BASE) && 
           (es.weight_full_addr < ROW_SIZE)));
    assert(!es.ifmap_valid ||
            ((es.psum_full_addr >= MMAP_PSUM_BASE) && 
             (es.psum_full_addr < MMAP_PSUM_BASE + COLUMN_SIZE)));
    return es;
}

PoolSignals
Sequencer::pull_pool() {
    return ps;
}


bool
Sequencer::done() {
    return feed->empty() && !es.ifmap_valid && !es.weight_valid && !ps.valid;
}
