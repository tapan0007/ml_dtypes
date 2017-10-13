import sys

from layers.layer                import DoBatching
from nets.densenet.densenet     import DenseNet169
from nets.resnet.resnet         import ResNet50
from schedule.scheduler         import Scheduler

#print sys.argv

PrintLevels = False
PrintSchedule = True
PrintDot = False
PrintLayers = False
DoBatching = False

for arg in sys.argv[1:]:
    if arg == "--densenet" or arg == "--dense":
        ntwk = DenseNet169()
    elif arg == "--resnet" or arg == "--res":
        ntwk = ResNet50()
    elif arg == "--print-layers":
        PrintLayers = True
    elif arg == "--print-levels":
        PrintLevels = True
    elif arg == "--no-print-sched":
        PrintSchedule = False
    elif arg == "--print-dot":
        PrintDot = True
    elif arg == "--batch" or arg == "--batching":
        DoBatching = True
    else:
        sys.stderr.write("Wrong argument: " + arg + "\n")
        sys.exit(1)

##################################################
assert(ntwk)
ntwk.construct()
scheduler = Scheduler()
scheduler.schedule(ntwk)
ntwk.rLevels(scheduler.gLevels())

### Printing
if PrintLayers:
    ntwk.printMe()
    print

if PrintLevels:
    print "By level"
    ntwk.printLevels()
    print

if PrintSchedule:
    print "By scheduling"
    ntwk.printSched()
    print

if PrintDot:
    print "Dot"
    ntwk.printDot()
    print


