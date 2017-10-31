

#include <stdio.h>

//Psuedo Code for simple NN : JDRTrivNet
/*
* Conv->MaxPool->Relu->Conv->softmax
*
* This is meant to showcase the logical equivalence of the instructions placed
* into the various instruction rams of the SP, PE, Pool and Activation engines,
* as well as shows the way the SB is filled in from the host or tonga dram
*
* THis will also show an annotation of the actual instructions for the tonga
* side by side.
*
* Simplifying assumptions for this :
* 1. no tiling
* 2. int8 mode -- thus PE is 128*128 (which can vary oer layer in real world)
*/

/************************************************************************/
//architecture specifications:
typedef uint64 sb_word_t;
typedef uint16 pe_array_weight_t;
typedef uint16 pe_array_fmap_t;

#define SB_SIZE (12*1024*1024) // HW SB_SIZE

static sb_word_t sb[SB_SIZE];

#define PE_WT_COL_SIZE (128) //size of array of pe weights in rows*columns
#define PE_WT_ROW_SIZE (128)
#define PE_FMAP_ROW_SIZE (128)
#define PE_FMAP_COL_SIZE (128) //size of array of pe fmaps in rows*columns
#define NUM_PSUM_BANK 4
#define SIZE_PSUM_BANK 256

static psum_array_t psum[NUM_PSUM_BANK][SIZE_PSUM_BANK];

/************************************************************************/

//define fixed compile time choices for SB memrory usage and layout
//based on specific net being used, but using a template of layout of regions.
#define SB_WT_ADDR 0
#define SB_WT_SIZE (1024*1024) //compile time - choose
#define SB_IF_MAP_ADDR (SB_WT_ADDR + SB_WT_SIZE)
#define SB_IF_MAP_SIZE 1024 // fixed based on image type
#define SB_CONV_1_RESULT_ADDR (SB_IF_MAP_ADDR + SB_IF_MAP_SIZE) // ofmap - if used SB to store result
#define SB_CONV_2_RESULT_ADDR  (SB_CONV_1_RESULT_ADDR + SIZE)// ofmap - if used SB to store result
#define SB_POOL_RESULT_ADDR (SB_CONV_2_RESULT_ADDR + SIZE)
#define SB_ACT_RESULT_ADDR (SB_POOL_RESULT_ADDR + SIZE)

#define PSUM_CONV_1_BANK 0 //select a bank for each layer
#define PSUM_CONV_2_BANK 1

/************************************************************************/

//params of JDR NET
#define JDR_WT_SET_1_SIZE 1024 //size of weights in conv layer 1 of JDRTrivNet
#define JDR_WT_SET_2_SIZE 1024 //size of weights in conv layer 2 of JDRTrivNet
#define JDR_IFMAP_SIZE (16*16) //we have encoded incoming image to fit
#define JDR_CONV_1_C 128 // this is the number channel/ifmap .. note: we chose to fit into all rows witout folding
#define JDR_CONV_1_RS (7*7) //size of filter for conv layer 1
#define JDR_CONV_2_RS (3*3) //size of filter for conv layer 2

//weights stored (staged ahead of time) in Tonga DRAM:
#define JDR_WT_SET_1_ADDR 0
#define JDR_WT_SET_2_ADDR (JDR_WT_SET_1_ADDR + JDR_WT_SET_1_SIZE)


void sp_func (void)
{
  //load weights into SB
  memcpy(TONGA_DRAM[JDR_WT_SET_1_ADDR], sb[SB_WT_ADDR], JDR_WT_SET_1_SIZE);

  //load PE, PSUM, Pool, ACT instr

  //load IFMAP
  memcpy(HOST_DRAM[JDR_WT_SET_2_ADDR], sb[IF_MAP_ADDR], JDR_IFMAP_SIZE);

  pe_psum_func_1();

  pool_func();

  act_func();

  //load new weights into SB
  //note - can overwrite the first set of weights here since not needed
  memcpy(TONGA_DRAM[JDR_WT_SET_2_ADDR], sb[SB_WT_ADDR], JDR_WT_SET_2_SIZE);
  pe_psum_func_2();

  //need to get the results out of psum and into the SB so host can collect.
  pool_func_final();

  //send finish signal back to HOST cpu
  //results are going to be in SB... host will collect from there.
  //softmax done in software on host
  SP_notify(done);
}

//implement conv layer 1
//input from SB, output back to SB
void pe_psum_func_1(void)
{
  pe_array_weight_t pe_wt[PE_WT_ROW_SIZE][PE_WT_COL_SIZE];
  pe_array_fmap_t pe_fmap[PE_FMAP_ROW_SIZE][PE_FMAP_COL_SIZE];


  //pick a bank to use in the PSUM for this Conv
  int bank = PSUM_CONV_1_BANK;
  // do we need to do this in instructions? is this needed?
  memset(psum[bank], 0, SIZE_PSUM_BANK]);

  //note: in real chip actually have to reload the fmaps and weights on each wave
  for (int i = 0; i < JDR_CONV_1_RS; i++) {
    memcpy(sb[SB_WT_ADDR + (i*PE_WT_SIZE)], pe_wt, PE_WT_ROW_SIZE * PE_WT_COL_SIZE);
    memcpy(sb[IF_MAP_ADDR], pe_fmap, JDR_IFMAP_SIZE);

    //this is the main matrix mult
    //mult_add(a, b, c) : c += a*b;
    mult_add(pe_wt, pe_fmap, psum_bank[bank])
  }
  //can copy this back to SB OR can leave in psum and access from there later...
  //memcpy(psum_bank[bank], sb[SB_CONV_1_RESULT_ADDR ], SIZE);
}

//implement conv layer 2
//input from SB, output back to SB
//NOTE: WEIGHTS WERE LOADED INTO SAME SB AREA AS FIRAT CONV LAYER
void  pe_psum_func_2(void)
{
  pe_array_weight_t pe_wt[PE_WT_ROW_SIZE][PE_WT_COL_SIZE];
  pe_array_fmap_t pe_fmap[PE_FMAP_ROW_SIZE][PE_FMAP_COL_SIZE];


  //pick a bank to use in the PSUM for this Conv
  int bank = PSUM_CONV_2_BANK;
  // do we need to do this in instructions? is this needed?
  memset(psum[bank], 0, SIZE_PSUM_BANK]);

  //note: in real chip actually have to reload the fmaps and weights on each wave
  for (int i = 0; i < JDR_CONV_2_RS; i++) {
    memcpy(sb[SB_WT_ADDR + (i*PE_WT_SIZE)], pe_wt, PE_WT_ROW_SIZE * PE_WT_COL_SIZE);
    memcpy(sb[IF_MAP_ADDR], pe_fmap, JDR_IFMAP_SIZE);

    //this is the main matrix mult
    //mult_add(a, b, c) : c += a*b;
    mult_add(pe_wt, pe_fmap, psum_bank[bank])
  }
  //can copy this back to SB OR can leave in psum and access from there later...
  //memcpy(psum_bank[bank], sb[SB_CONV_1_RESULT_ADDR ], SIZE);
}

//pooling - input comes from Psum (could be SB), output back to SB (could be output to psum)
void pool_func (void)
{
  int bank = PSUM_CONV_1_BANK;
  //maxpool(a,b): b=maxpool(a);
  maxpool(psum[bank], sb[SB_POOL_RESULT_ADDR])
}

//activation -- input from SB (could also be psum), output back to SB (could be Psum)
//Relu()....input comes from SB after pool layer
void act_func (void)
{
  //matrix based relu(a,b): b=relu(a);
  relu(sb[SB_POOL_RESULT_ADDR], sb[SB_ACT_RESULT_ADDR]);
}

void pool_func_final (void)
{
  //identity operation in pool engine
  //pool_identity(a,b): b=a;
  pool_identity(psum[PSUM_CONV_2_BANK], sb[SB_CONV_2_RESULT_ADDR]);
}
