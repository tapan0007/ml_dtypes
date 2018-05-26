from layeropt_utils import data_type_to_item_sz

class BatchSlot:
    slot_count = 0
    def __init__(self, size):
        self.size = size
        self.slot = BatchSlot.slot_count
        BatchSlot.slot_count += 1
        self.batch_item = -1
    def assign(self, batch_item):
        if (self.batch_item != -1):
            raise RuntimeError("Cannot assign to an occupied slot %d, which contains batch item %d"%(self.slot, self.batch_item))
        print("assigning slot %d to batch item %d"%(self.slot, batch_item))
        self.batch_item = batch_item
    def move_from(self, slot):
        print("moving slot %d item to slot %d for batch item %d"%(slot.slot, self.slot, slot.batch_item))
        self.assign(slot.batch_item)
        slot.batch_item = -1
    def show(self):
        print("%d:%d (%d)"%(self.slot, self.batch_item, self.size), end=" ")

class BatchMachine:
    def __init__(self, batch_size, data_type):
        self.item_sz         = data_type_to_item_sz(data_type)
        self.data_type       = data_type
        self.N = batch_size  # can only accept 16 at the moment
        assert(batch_size == 16)
        assert(self.item_sz == 2)

        # below are sizing for the recursive walk demonstration
        self.pairup_at_layer =         [2, 10, 20, 30, -1]
        self.pairup_at_layer_item_sz = [6272*self.item_sz,  3136*self.item_sz,  1568*self.item_sz,  784*self.item_sz,   0]
        # initialize residue and batch slots    
        self.residue_slot = BatchSlot(self.pairup_at_layer_item_sz[0])
        self.batch_slots_per_level = []
        for i in range(len(self.pairup_at_layer_item_sz)-1):
            self.batch_slots_per_level.append([BatchSlot(self.pairup_at_layer_item_sz[i]) for j in range(2**i)])

        # below are sizing for Layeropt2 batching scheme
        bias_sz = 512
        ofmap_sz_56x56x64   = 56 * 56 * self.item_sz            # = 6272
        ofmap_sz_56x56x256  = 56 * 56 * 256 * self.item_sz // 128    # = 12544
        ofmap_sz_28x28x512  = 28 * 28 * 512 * self.item_sz // 128    # = 6272
        ofmap_sz_14x14x1024 = 14 * 14 * 1024 * self.item_sz // 128   # = 3136
        ofmap_sz_7x7x2048   = 7 * 7 * 2048 * self.item_sz // 128     # = 1568
        partialbatch_sz     = ofmap_sz_7x7x2048 * 9 + ofmap_sz_14x14x1024 * 5 + ofmap_sz_28x28x512 * 3 + ofmap_sz_56x56x64 * 1 + ofmap_sz_56x56x64 * 1 # includes extra space to prevent overwrite
        partialbatch_sz_8   = ofmap_sz_7x7x2048 * 9 + ofmap_sz_14x14x1024 * 8
        partialbatch_sz_16  = ofmap_sz_7x7x2048 * 16
        
        #                         batch=1, pre-pairup=X      batch=2                     batch=2, pre-pairup=T       batch=4                     batch=4, pre-pairup=T       batch=8                     batch=8, pre-pairup=T         batch=16
        self.sb_bias_sz =         [bias_sz,                  bias_sz,                    bias_sz,                    bias_sz,                    bias_sz,                    bias_sz,                    bias_sz,                      bias_sz]
        self.sb_partialbatch_sz = [partialbatch_sz,          partialbatch_sz,            partialbatch_sz,            partialbatch_sz,            partialbatch_sz,            partialbatch_sz_8,          partialbatch_sz_8,            partialbatch_sz_16]
        self.sb_weights_sz =      [7*7*64*self.item_sz,      3*3*64*self.item_sz,        3*3*64*2*self.item_sz,      3*3*128*self.item_sz,       3*3*256*2*self.item_sz,     3*3*256*2*self.item_sz,     3*3*512*4*self.item_sz,       3*3*512*4*self.item_sz]
        self.sb_scratch_sz =      [112*112*self.item_sz,     ofmap_sz_56x56x256*2,       28*28*self.item_sz*3,       28*28*self.item_sz*5,       14*14*2*self.item_sz*5,     14*14*2*self.item_sz*9,     7*7*4*self.item_sz*12,        7*7*4*self.item_sz*20 ]

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
        self.sb_partialbatch_start[8] = ofmap_sz_7x7x2048 * 9 
        self.sb_partialbatch_start[4] = self.sb_partialbatch_start[8] + ofmap_sz_14x14x1024 * 5
        self.sb_partialbatch_start[2] = self.sb_partialbatch_start[4] + ofmap_sz_28x28x512 * 3      # MaxPool output, followed by conv layer outputing to scratch 
        self.sb_partialbatch_start[1] = self.sb_partialbatch_start[2] + ofmap_sz_56x56x64 * 1       # IFMAP input space, 1st layer output goes to scratch

    def show_slots(self):
        for i in range(len(self.pairup_at_layer_item_sz)-1):
            for j in self.batch_slots_per_level[i]:
                j.show()
        print("")            

    def total_slots_size(self):
        total = 0
        for i in range(len(self.pairup_at_layer_item_sz)-1):
            for j in self.batch_slots_per_level[i]:
                total += j.size
        return total

    def check_sb_usage(self):
        # check SB usage
        for i in range(len(self.sb_bias_sz)):
            #total_sb = self.sb_bias_sz[i] + self.sb_ifmaps_sz[i] + self.sb_partialbatch_sz[i] + self.sb_weights_sz[i] + self.sb_residue_sz[i] + self.sb_scratch_sz[i]
            total_sb = self.sb_bias_sz[i] + self.sb_partialbatch_sz[i] + self.sb_weights_sz[i] + self.sb_scratch_sz[i]
            print("Total SB usage %d (headroom %d, batch_slots check %d)"%(total_sb, 96*1024 - total_sb, self.total_slots_size()))

    def process_subbatch(self, batch_items, level, left):
        num = len(batch_items)
        mid = num//2
        print(batch_items)
        level_adjusted = max(0, len(self.pairup_at_layer) - 1 - level)
        if (num>1):
            self.process_subbatch(batch_items[0:mid], level+1, True)
            self.process_subbatch(batch_items[mid:], level+1, False)
            print("Level %d: processing batch items %s until shrink point %d"%(level, str(tuple(batch_items)), self.pairup_at_layer[level_adjusted]))
            if (num == 2):
                if left:
                    self.batch_slots_per_level[level_adjusted][0].move_from(self.residue_slot)
                    self.batch_slots_per_level[level_adjusted][1].move_from(self.batch_slots_per_level[0][0])
            elif (num == 4):           
                if left:
                    self.batch_slots_per_level[level_adjusted][0].move_from(self.residue_slot)
                    self.batch_slots_per_level[level_adjusted][1].move_from(self.batch_slots_per_level[0][0])
                    self.batch_slots_per_level[level_adjusted][2].move_from(self.batch_slots_per_level[1][0])
                    self.batch_slots_per_level[level_adjusted][3].move_from(self.batch_slots_per_level[1][1])
            elif (num == 8):            
                if left:
                    self.batch_slots_per_level[level_adjusted][0].move_from(self.residue_slot)
                    self.batch_slots_per_level[level_adjusted][1].move_from(self.batch_slots_per_level[0][0])
                    self.batch_slots_per_level[level_adjusted][2].move_from(self.batch_slots_per_level[1][0])
                    self.batch_slots_per_level[level_adjusted][3].move_from(self.batch_slots_per_level[1][1])
                    self.batch_slots_per_level[level_adjusted][4].move_from(self.batch_slots_per_level[2][0])
                    self.batch_slots_per_level[level_adjusted][5].move_from(self.batch_slots_per_level[2][1])
                    self.batch_slots_per_level[level_adjusted][6].move_from(self.batch_slots_per_level[2][2])
                    self.batch_slots_per_level[level_adjusted][7].move_from(self.batch_slots_per_level[2][3])
        elif (num == 1):
            print("Level %d: processing batch item %d until shrink point %d"%(level, batch_items[0], self.pairup_at_layer[level_adjusted]))
            if left:            
                self.batch_slots_per_level[0][0].assign(batch_items[0])
            else:            
                self.residue_slot.assign(batch_items[0])
        self.show_slots()                

# Main program
if __name__ == "__main__":
    # process batch
    batch_machine = BatchMachine(16,'float16')
    batch_machine.check_sb_usage()
    batch_machine.process_subbatch(list(range(16)), 0, False)

