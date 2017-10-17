import sys
from utils.printers             import Printer
import layers.layer

#from layers.layer               import rDoBatching

from nets.densenet.densenet     import DenseNet169
from nets.resnet.resnet         import ResNet50
from schedule.scheduler         import Scheduler

#print sys.argv

PrintLevels = False
PrintSchedule = True
PrintDot = False
PrintLayers = False
__DoBatching = False

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
        __DoBatching = True
    else:
        sys.stderr.write("Wrong argument: " + arg + "\n")
        sys.exit(1)

##################################################
assert(ntwk)
ntwk.rDoBatching(__DoBatching)
ntwk.construct()
scheduler = Scheduler()
scheduler.Schedule(ntwk)
ntwk.rLevels(scheduler.gLevels())

##################################################
printer = Printer(ntwk)

### Printing
if PrintLayers:
    printer.printNetwork()
    print

if PrintLevels:
    print("By level")
    printer.printLevels()
    print

if PrintSchedule:
    print("By scheduling")
    printer.printSched()
    print

if PrintDot:
    print("Dot")
    printer.printDot()
    print


