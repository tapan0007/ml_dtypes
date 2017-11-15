#include <assert.h>
#include "sequencer.h"
#include "string.h"
#include "tpb_isa.h"
#include "io.h"


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
    Sequencer *seq = (Sequencer *)v_seq;
    void *i_ptr;
    int i_n, i_c, i_h, i_w;
    size_t word_size;

    /* load io_mmap */
    i_ptr = seq->mem->io_mmap(args.fname, i_n, i_c, i_h, i_w, word_size);
    seq->mem->bank_mmap(args.address, i_ptr, i_c, i_h * i_w * word_size);
    delete [] (char *)i_ptr;
}

/*------------------------------------
 * RdFilterArgs
 *------------------------------------ */
template<>
void  DynamicInstruction<SIM_RDFILTER>::execute(void *v_seq) {
    Sequencer *seq = (Sequencer *)v_seq;
    void *f_ptr;
    int r,s,t,u;
    int w_c, w_m, w_r, w_s;
    size_t word_size;

    /* load io_mmap */
    f_ptr = seq->mem->io_mmap(args.fname, r, s, t, u, word_size);
    seq->mem->swap_axes(f_ptr, r, s, t, u, word_size);
    w_c = s; // for swamp, M now corresponds to C
    w_m = r; // for swamp, C now corresponds to M
    w_r = t;
    w_s = u;
    seq->mem->bank_mmap(args.address, f_ptr, w_c, w_m * w_r * w_s * word_size);
    delete [] (char *)f_ptr;
}

/*------------------------------------
 * WrOfmapArgs
 *------------------------------------ */
template<>
void  DynamicInstruction<SIM_WROFMAP>::execute(void *v_seq) {
    Sequencer *seq = (Sequencer *)v_seq;
    void *o_ptr;
    ARBPRECTYPE dtype = (ARBPRECTYPE)(args.dtype);

    o_ptr = seq->mem->sbuffer_bank_munmap(args.address, args.dims[1],
            args.dims[2] * args.dims[3] * sizeofArbPrecType(dtype));
    seq->mem->io_write(args.fname, o_ptr, args.dims[0], args.dims[1],
            args.dims[2], args.dims[3], dtype);
    free(o_ptr);
}


/*------------------------------------
 * LdWeightsInstruction
 *------------------------------------ */
template<>
void  DynamicInstruction<LDWEIGHTS>::execute(void *v_seq) {
    Sequencer *seq = (Sequencer *)v_seq;
    uint64_t num_cols = args.x_num;
    ARBPRECTYPE dtype = (ARBPRECTYPE) args.dquant.dequant_data_type;
    size_t dsize = sizeofArbPrecType(dtype);
    assert(num_cols <= SZ(COLUMN_BITS));

    int weight_nums[]   = {args.x_num};
    int weight_steps[]  = {args.x_step};
    seq->weight_pit.init(weight_nums, weight_steps);
    /* load weights into ofmaps backwards! */
    seq->weight_base = args.start_addr + (args.x_num * args.x_step - 1) * dsize;
    seq->weight_clamp_countdown = num_cols;
    seq->raw_signal = false;

    seq->es.weight_valid = true;
    seq->es.weight_dtype = (ARBPRECTYPE) args.dquant.dequant_data_type;
    //seq->es.weight_addr.sys = args.address;
    seq->es.weight_clamp = (num_cols == 1);
    seq->es.row_valid = true;
    seq->es.row_countdown = args.num_row_partitions;
    seq->es.weight_addr.sys = seq->weight_base;

}

/*------------------------------------
 * Convolve
 *------------------------------------ */
template<>
void  DynamicInstruction<MATMUL>::execute(void *v_seq) {
    Sequencer *seq = (Sequencer *)v_seq;
    /* ifmap setup */
    seq->es.row_countdown = args.num_row_partitions;
    seq->es.row_valid = true;
    seq->es.weight_toggle = args.toggle_weight;

    int ifmap_nums[]   = {args.fmap_x_num, args.fmap_y_num};
    int ifmap_steps[]  = {args.fmap_x_step, args.fmap_y_step};
    seq->ifmap_pit.init(ifmap_nums, ifmap_steps);
    seq->es.ifmap_addr.sys    = args.fmap_start_addr;
    seq->es.ifmap_dtype = (ARBPRECTYPE) 
        args.dquant.dequant_data_type;
    seq->ifmap_base = args.fmap_start_addr;
    seq->raw_signal = false;

    /* psum setup */
    seq->es.column_countdown = args.num_column_partitions;
    seq->es.column_valid = true;
    seq->es.psum_start = args.start_tensor_calc;
    seq->es.psum_stop = args.stop_tensor_calc;
    seq->es.psum_addr.sys = args.psum_start_addr;

    /* signals that will stay constant for entire convolution */
    /* padding setup */
    seq->pad[N] = args.n_pad;
    seq->pad[S] = args.s_pad;
    seq->pad[E] = args.e_pad;
    seq->pad[W] = args.w_pad;
    /* pifmap is padded ifmap */
    int ew_pad = args.e_pad + args.w_pad;
    int ns_pad = args.n_pad + args.s_pad;
    int pifmap_nums[]  = {args.fmap_x_num + ew_pad,  args.fmap_y_num + ns_pad};
    int pifmap_steps[] = {args.fmap_x_step + ew_pad, args.fmap_y_step + ns_pad};
    seq->pifmap_pit.init(pifmap_nums, pifmap_steps);
    seq->es.pad_valid = seq->pad_valid();
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
    seq->ps.in_dtype  = (ARBPRECTYPE)args.in_dtype;
    seq->ps.out_dtype = (ARBPRECTYPE)args.out_dtype;
    seq->ps.src_addr.sys = args.src_start_addr;
    seq->ps.start = true;
    seq->ps.stop = (pool_func == IDENTITY_POOL) ||
        (args.src_x_num + args.src_y_num == 2);
    seq->ps.dst_addr.sys = args.dst_start_addr;
    seq->ps.avg_count = args.src_x_num * args.src_y_num;
    seq->ps.countdown = args.num_partitions;

    seq->pool_src_base = args.src_start_addr;
    seq->pool_dst_base = args.dst_start_addr;


    int pool_src_nums[]   = {args.src_x_num, args.src_y_num};
    int pool_src_steps[]  = {args.src_x_step, args.src_y_step};
    seq->pool_src_pit.init(pool_src_nums, pool_src_steps);

    int pool_str_nums[]   = {args.str_x_num, args.str_y_num};
    int pool_str_steps[]  = {args.str_x_step, args.str_y_step};
    seq->pool_str_pit.init(pool_str_nums, pool_str_steps);

    int pool_dst_nums[]   = {args.dst_x_num,  args.dst_y_num};
    int pool_dst_steps[]  = {args.dst_x_step, args.dst_y_step};
    seq->pool_dst_pit.init(pool_dst_nums, pool_dst_steps);

    /* emit one result for every stride... if not identity pool */
    assert((pool_func == IDENTITY_POOL) ||
            (args.dst_x_num * args.dst_y_num ==
             args.str_x_num * args.str_y_num));
}

/*------------------------------------
 * Activation
 *------------------------------------ */
template<>
void  DynamicInstruction<ACTIVATION>::execute(void *v_seq) {
    Sequencer *seq = (Sequencer *)v_seq;
    ACTIVATIONFUNC act_func = (ACTIVATIONFUNC)args.activation_func;
    seq->as.valid = true;
    seq->as.func  = act_func;
    seq->as.in_dtype  = (ARBPRECTYPE)args.in_data_type;
    seq->as.out_dtype = (ARBPRECTYPE)args.out_data_type;
    seq->as.src_addr.sys = args.src_start_addr;
    seq->as.dst_addr.sys = args.dst_start_addr;
    seq->as.countdown = args.num_partitions;

    seq->act_src_base = args.src_start_addr;
    seq->act_dst_base = args.dst_start_addr;
    assert((args.in_data_type == args.out_data_type) && 
            "type conversion not supported yet");


    int act_src_nums[]   = {args.src_x_num_elements, args.src_y_num_elements};
    int act_src_steps[]  = {args.src_x_step, args.src_y_step};
    seq->act_src_pit.init(act_src_nums, act_src_steps);

    int act_dst_nums[]   = {args.dst_x_num_elements,  args.dst_y_num_elements};
    int act_dst_steps[]  = {args.dst_x_step, args.dst_y_step};
    seq->act_dst_pit.init(act_dst_nums, act_dst_steps);

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
        ps.valid || as.valid;
    if (es.ifmap_valid || ps.valid || as.valid) {
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
Sequencer::pad_valid() {
    /* checks range, using outer ring of padding */
    uint8_t r = pifmap_pit[1];
    uint8_t c = pifmap_pit[0];
    int prange[] = {c, r};
    int irange[] = {c-pad[W], r-pad[N]};
    bool in_fmap_range = ifmap_pit.in_range(irange);
    bool in_pad_range   = pifmap_pit.in_range(prange);
    return !pifmap_pit.eop() && !in_fmap_range && in_pad_range;
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
        pifmap_pit.increment();
    }
    if (es.ifmap_valid) {
        ifmap_pit.increment();
        pifmap_pit.increment();
    }
    if (es.weight_valid) {
        weight_pit.increment();
    }

    /* UPDATE SIGNALS */
    /* IFMAP/PAD */
    /* is the pad/ifmap valid or are we done? */
    es.pad_valid = pad_valid();
    if (!es.pad_valid && pifmap_pit.eop()) {
        /* clear state */
        es.ifmap_valid = false;
        es.psum_start = false;
        es.psum_stop = false;
    } else {
        es.ifmap_valid = !es.pad_valid;
    }

    /* figured out pad/ifmap valid, now compute addresses */
    if (es.ifmap_valid) {
        es.ifmap_addr.sys = ifmap_base +
            ifmap_pit.coordinates() *
            sizeofArbPrecType((ARBPRECTYPE)es.ifmap_dtype);
    }
    /* Sending an ifmap, must be getting out ofmap! */
    if (es.ifmap_valid || es.pad_valid) {
        /* FIXME - this is the psum granularity, FIX */
        es.psum_addr.sys += SZ(PSUM_ENTRY_BITS);
    }

    /*  WEIGHT */
    /* always toggle down weight */
    COND_SET(es.weight_toggle, false);
    if (es.weight_valid) {
        if (weight_pit.eop()) {
            assert(es.weight_clamp);
            es.weight_clamp = false;
            es.weight_valid = false;
        } else if (weight_pit.last()) {
            es.weight_clamp = true;
        }
        if (es.weight_valid) {
            es.weight_addr.sys = weight_base -
                weight_pit.coordinates() *
                sizeofArbPrecType((ARBPRECTYPE)es.weight_dtype);
        }
    }
}

namespace rollover {
    template <typename T> void increment(T& value)
    {
        ++value;
    }

    template <> void increment<bool>(bool& value)
    {
        value = !value;
    }
};

#define ROLLOVER(CNT, NUM, ROLLOVER) \
    if (CNT >= NUM) { \
        CNT = 0; \
        rollover::increment(ROLLOVER); \
    }


/* sub function of step - to step the poolsignal */

void
Sequencer::step_poolsignal() {
    if (!ps.valid) {
        return;
    }


    /* eventually want to support quantization here */
    size_t src_dsize = sizeofArbPrecType(ps.in_dtype);
    size_t dst_dsize = sizeofArbPrecType(ps.out_dtype);

    pool_src_pit.increment();


    /* roll over dst counters */
    if (ps.func == IDENTITY_POOL || pool_src_pit.eop()) {
        pool_dst_pit.increment();
        pool_str_pit.increment();
        if (ps.func != IDENTITY_POOL) {
            pool_src_pit.reset();
        }
    }

    /* stopped last cycle, start anew */
    if (ps.func == IDENTITY_POOL) {
        ps.start = true;
        ps.stop = true;
    } else {
        ps.start = ps.stop; /* stopped last cycle, start anew */
        ps.stop = pool_src_pit.last();
    }

    assert((ps.func == IDENTITY_POOL) || 
            (pool_str_pit.eop() == pool_dst_pit.eop()));

    /* calculate address based on settings */
    if (ps.valid) {
        ps.src_addr.sys = pool_src_base + src_dsize *
            (pool_str_pit.coordinates() + /* tile */
             pool_src_pit.coordinates()); /* offset in tile */
    }
    if (ps.start) {
        ps.dst_addr.sys = pool_dst_base + dst_dsize * 
            pool_dst_pit.coordinates();
    }
    if (pool_dst_pit.eop()) {
        ps.valid = false;
        return;
    }
}

/* sub function of step - to step the poolsignal */
void
Sequencer::step_actsignal() {
    if (!as.valid) {
        return;
    }
    /* eventually want to support quantization here */
    size_t src_dsize = sizeofArbPrecType(as.in_dtype);
    size_t dst_dsize = sizeofArbPrecType(as.out_dtype);

    act_src_pit.increment();
    act_dst_pit.increment();

    /* calculate address based on settings */
    if (as.valid) {
        as.src_addr.sys = act_src_base + src_dsize * act_src_pit.coordinates();
        as.dst_addr.sys = act_dst_base + dst_dsize * act_dst_pit.coordinates();
    }
    if (act_dst_pit.eop()) {
        as.valid = false;
    }
}

void
Sequencer::step() {
    /* empty the instruction queue */
    Instruction *inst = NULL;
    if (!synch()) {
        if (!feed->empty()) {
            uint64_t *raw_inst = (uint64_t *)feed->front();
            switch (((TPB_CMD_HEADER *)raw_inst)->opcode) {
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
                case ACTIVATION_OPC:
                    inst = new DynamicInstruction<ACTIVATION>(
                            *((ACTIVATION *) (raw_inst)));
                    break;
                default:
                    assert(0);
            }
            inst->execute(this);
            feed->pop();
            delete inst;
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
    step_actsignal();

    clock++;
}


EdgeSignals Sequencer::pull_edge() {
// IF: to avoid warning: comparison is always true due to limited range of data type [-Wtype-limits]
    assert(!es.ifmap_valid || (
#if MMAP_SB_BASE
            (es.ifmap_addr.sys >= MMAP_SB_BASE) &&
#endif
             (es.ifmap_addr.sys < SZ(ROW_SIZE_BITS))));
    assert(!es.weight_valid || (
#if MMAP_SB_BASE
          (es.weight_addr.sys >= MMAP_SB_BASE) &&
#endif
           (es.weight_addr.sys < SZ(ROW_SIZE_BITS))));
    assert(!es.ifmap_valid ||
            ((es.psum_addr.sys >= MMAP_PSUM_BASE) &&
             (es.psum_addr.sys < MMAP_PSUM_BASE + SZ(COLUMN_SIZE_BITS))));
    return es;
}

PoolSignals
Sequencer::pull_pool() {
    return ps;
}

ActivateSignals
Sequencer::pull_activate() {
    return as;
}


bool
Sequencer::done() {
    return feed->empty() && !es.ifmap_valid && !es.weight_valid && 
        !ps.valid && !as.valid;
}
