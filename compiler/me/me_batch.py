"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

"""SB data map for batching
"""

from me_utils import data_type_to_item_sz

class BatchSBDataMap:
    def __init__(self, batch_size, data_type):
        self.item_sz         = data_type_to_item_sz(data_type)
        self.data_type       = data_type
        self.N = batch_size  # can only accept 16 at the moment
        assert(batch_size == 16)
        assert(self.item_sz == 2)

        # below are sizing for ResNet50 batching scheme
        bias_sz = 512
        #self.ofmap_sz_56x56x64   = 56 * 56 * self.item_sz            # = 6272
        self.ofmap_sz_55x55x64   = 55 * 55 * self.item_sz            # = 6050
        #self.ofmap_sz_56x56x256  = 56 * 56 * 256 * self.item_sz // 128    # = 12544
        self.ofmap_sz_55x55x256  = 55 * 55 * 256 * self.item_sz // 128    # = 12100
        self.ofmap_sz_28x28x512  = 28 * 28 * 512 * self.item_sz // 128    # = 6272
        self.ofmap_sz_14x14x1024 = 14 * 14 * 1024 * self.item_sz // 128   # = 3136
        self.ofmap_sz_7x7x2048   = 7 * 7 * 2048 * self.item_sz // 128     # = 1568
        #partialbatch_sz     = self.ofmap_sz_7x7x2048 * 12 + self.ofmap_sz_14x14x1024 * 5 + self.ofmap_sz_28x28x512 * 3 + self.ofmap_sz_56x56x64 * 1 + self.ofmap_sz_56x56x64 * 1 # includes extra space to prevent overwrite
        partialbatch_sz     = self.ofmap_sz_7x7x2048 * 12 + self.ofmap_sz_14x14x1024 * 5 + self.ofmap_sz_28x28x512 * 3 + self.ofmap_sz_55x55x64 * 1 + self.ofmap_sz_55x55x64 * 1 # includes extra space to prevent overwrite
        partialbatch_sz_8   = self.ofmap_sz_7x7x2048 * 12 + self.ofmap_sz_14x14x1024 * 8
        partialbatch_sz_16  = self.ofmap_sz_7x7x2048 * 16
        
        #                         batch=1, pre-pairup=X      batch=2                     batch=2, pre-pairup=T       batch=4                     batch=4, pre-pairup=T       batch=8                     batch=8, pre-pairup=T         batch=16
        self.sb_bias_sz =         [bias_sz,                  bias_sz,                    bias_sz,                    bias_sz,                    bias_sz,                    bias_sz,                    bias_sz,                      bias_sz]
        self.sb_partialbatch_sz = [partialbatch_sz,          partialbatch_sz,            partialbatch_sz,            partialbatch_sz,            partialbatch_sz,            partialbatch_sz_8,          partialbatch_sz_8,            partialbatch_sz_16]
        self.sb_weights_sz =      [7*7*64*self.item_sz,      3*3*64*self.item_sz,        3*3*64*2*self.item_sz,      3*3*128*self.item_sz,       3*3*256*2*self.item_sz,     3*3*256*2*self.item_sz,     3*3*512*4*self.item_sz,       3*3*512*4*self.item_sz]
        #self.sb_scratch_sz =      [112*112*self.item_sz,     ofmap_sz_56x56x256*2,       28*28*self.item_sz*3,       28*28*self.item_sz*5,       14*14*2*self.item_sz*5,     14*14*2*self.item_sz*9,     7*7*4*self.item_sz*12,        7*7*4*self.item_sz*20 ]
        #self.sb_scratch_sz =      [112*112*self.item_sz,     ofmap_sz_56x56x256*2,       28*28*self.item_sz*2,       28*28*self.item_sz*4,       14*14*2*self.item_sz*4,     14*14*2*self.item_sz*8,     7*7*4*self.item_sz*8,        7*7*4*self.item_sz*16]
        self.sb_scratch_sz =      [112*112*self.item_sz,     56*56*2*self.item_sz*2,      28*28*self.item_sz*2,       28*28*self.item_sz*4,       14*14*2*self.item_sz*4,     14*14*2*self.item_sz*8,     7*7*4*self.item_sz*8,        7*7*4*self.item_sz*16]
        # (using 56*56 in sb_scratch_sz[1] to accomodate one-layer test that mimics ResNet50 input layer, ie. 3-1conv0_padvalid_wave)

        # Set of sizes for each current batch level and "pre-pairup" flag.
        # "pairup" is the region or boundary where OFMAP shrinks by 1/4 and partial-batch count doubles.
        # "pre-pairup" is the region just before "pairup" where OFMAP shrinks by 1/4 but partial-batch count has not doubled.
        # When selecting the set for OFMAP and at the pairup fused layer (last op is join or is fork), use next fused layer's batch level
        self.sb_size_set_index = {}
        self.sb_size_set_index[(1, False)] = 0
        self.sb_size_set_index[(1, True)] = 0
        self.sb_size_set_index[(2, False)] = 1   
        self.sb_size_set_index[(2, True)] = 2
        self.sb_size_set_index[(4, False)] = 3
        self.sb_size_set_index[(4, True)] = 4
        self.sb_size_set_index[(8, False)] = 5
        self.sb_size_set_index[(8, True)] = 6
        self.sb_size_set_index[(16, False)] = 7
        self.sb_size_set_index[(16, True)] = 7

        # start addresses for each batch level
        self.sb_partialbatch_start = {}
        self.sb_partialbatch_start[16] = 0
        self.sb_partialbatch_start[8] = self.ofmap_sz_7x7x2048 * 12 
        self.sb_partialbatch_start[4] = self.sb_partialbatch_start[8] + self.ofmap_sz_14x14x1024 * 5
        self.sb_partialbatch_start[2] = self.sb_partialbatch_start[4] + self.ofmap_sz_28x28x512 * 3      # MaxPool output, followed by conv layer outputing to scratch 
        #self.sb_partialbatch_start[1] = self.sb_partialbatch_start[2] + self.ofmap_sz_56x56x64 * 1       # IFMAP input space, 1st layer output goes to scratch
        self.sb_partialbatch_start[1] = self.sb_partialbatch_start[2] + self.ofmap_sz_55x55x64 * 1       # IFMAP input space, 1st layer output goes to scratch

    def reevaluate_set_select(self, fmap_sb_usage_size):
        if fmap_sb_usage_size <= self.ofmap_sz_7x7x2048:
            self.sb_size_set_index[(1, False)] = 7
            self.sb_size_set_index[(1, True)] = 7
            self.sb_partialbatch_start[1] = self.sb_partialbatch_start[16]
        elif fmap_sb_usage_size <= self.ofmap_sz_14x14x1024:
            self.sb_size_set_index[(1, False)] = 5
            self.sb_size_set_index[(1, True)] = 6
            self.sb_partialbatch_start[1] = self.sb_partialbatch_start[8]
        elif fmap_sb_usage_size <= self.ofmap_sz_28x28x512:
            self.sb_size_set_index[(1, False)] = 3
            self.sb_size_set_index[(1, True)] = 4
            self.sb_partialbatch_start[1] = self.sb_partialbatch_start[4]
        elif fmap_sb_usage_size <= self.ofmap_sz_55x55x256:
            self.sb_size_set_index[(1, False)] = 1
            self.sb_size_set_index[(1, True)] = 2
            self.sb_partialbatch_start[1] = self.sb_partialbatch_start[2]

    def check_sb_usage(self):
        for i in range(len(self.sb_bias_sz)):
            #total_sb = self.sb_bias_sz[i] + self.sb_ifmaps_sz[i] + self.sb_partialbatch_sz[i] + self.sb_weights_sz[i] + self.sb_residue_sz[i] + self.sb_scratch_sz[i]
            total_sb = self.sb_bias_sz[i] + self.sb_partialbatch_sz[i] + self.sb_weights_sz[i] + self.sb_scratch_sz[i]
            print("Total SB usage %d (headroom %d)"%(total_sb, 96*1024 - total_sb))

if __name__ == "__main__":
    # process batch
    batch_machine = BatchMachine(16,'float16')
    batch_machine.check_sb_usage()

