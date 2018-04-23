import layeropt 

N=16
pairup_at_layer =         [2, 10, 20, 30, -1]
pairup_at_layer_item_sz = [3136,  3136,  1568,  784,   0]
sb_bias_sz =         [1024,  1024,  1024,  1024,  1024]
sb_ifmaps_sz =       [4096,  25088, 12320, 6272,  3136]
sb_partialbatch_sz = [39200, 39200, 39200, 39200, 39200]
sb_weights_sz =      [6272,  2304,  10240, 36864, 36864]
sb_residue_sz =      [0,     12320, 6272,  3136,  1568]
sb_scratch_sz =      [25088, 12320, 6272,  3136,  1568]

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

residue_slot = BatchSlot(pairup_at_layer_item_sz[0])
batch_slots_per_level = []
for i in range(len(pairup_at_layer_item_sz)-1):
    batch_slots_per_level.append([BatchSlot(pairup_at_layer_item_sz[i]) for j in range(2**i)])

for i in range(len(pairup_at_layer_item_sz)-1):
    total_sb = sb_bias_sz[i] + sb_ifmaps_sz[i] + sb_partialbatch_sz[i] + sb_weights_sz[i] + sb_residue_sz[i] + sb_scratch_sz[i]
    print("Total SB usage %d"%total_sb)

def show_slots():
    for i in range(len(pairup_at_layer_item_sz)-1):
        for j in batch_slots_per_level[i]:
            j.show()
    print("")            

def process_subbatch(batch_items, level, left):
    num = len(batch_items)
    mid = num//2
    print(batch_items)
    level_adjusted = max(0, len(pairup_at_layer) - 1 - level)
    if (num>1):
        process_subbatch(batch_items[0:mid], level+1, True)
        process_subbatch(batch_items[mid:], level+1, False)
        print("Level %d: processing batch items %s until shrink point %d"%(level, str(tuple(batch_items)), pairup_at_layer[level_adjusted]))
        if (num == 2):
            if left:
                batch_slots_per_level[level_adjusted][0].move_from(residue_slot)
                batch_slots_per_level[level_adjusted][1].move_from(batch_slots_per_level[0][0])
        elif (num == 4):           
            if left:
                batch_slots_per_level[level_adjusted][0].move_from(residue_slot)
                batch_slots_per_level[level_adjusted][1].move_from(batch_slots_per_level[0][0])
                batch_slots_per_level[level_adjusted][2].move_from(batch_slots_per_level[1][0])
                batch_slots_per_level[level_adjusted][3].move_from(batch_slots_per_level[1][1])
        elif (num == 8):            
            if left:
                batch_slots_per_level[level_adjusted][0].move_from(residue_slot)
                batch_slots_per_level[level_adjusted][1].move_from(batch_slots_per_level[0][0])
                batch_slots_per_level[level_adjusted][2].move_from(batch_slots_per_level[1][0])
                batch_slots_per_level[level_adjusted][3].move_from(batch_slots_per_level[1][1])
                batch_slots_per_level[level_adjusted][4].move_from(batch_slots_per_level[2][0])
                batch_slots_per_level[level_adjusted][5].move_from(batch_slots_per_level[2][1])
                batch_slots_per_level[level_adjusted][6].move_from(batch_slots_per_level[2][2])
                batch_slots_per_level[level_adjusted][7].move_from(batch_slots_per_level[2][3])
    elif (num == 1):
        print("Level %d: processing batch item %d until shrink point %d"%(level, batch_items[0], pairup_at_layer[level_adjusted]))
        if left:            
            batch_slots_per_level[0][0].assign(batch_items[0])
        else:            
            residue_slot.assign(batch_items[0])
    show_slots()                

process_subbatch(list(range(N)), 0, False)
