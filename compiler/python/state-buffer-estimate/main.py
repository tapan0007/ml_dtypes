import sys
assert(sys.version_info.major >= 3)

from utils.printers             import Printer
from utils.debug                import breakFunc

import layers.layer
from arch.arch import Arch
from memmgr.statebufmgr         import StateBufferMgr

#from layers.layer               import rDoBatching

from schedule.scheduler         import Scheduler

#print sys.argv

PrintLevels = False
PrintSchedule = True
PrintDot = False
PrintLayers = False
__DoBatching = False
DenseNet = False
ResNet = False
UseRelu = False

for arg in sys.argv[1:]:
    if arg == "--densenet" or arg == "--dense":
        DenseNet = True
    elif arg == "--resnet" or arg == "--res":
        ResNet = True
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
    elif arg == "--relu":
        UseRelu = True
    else:
        sys.stderr.write("Wrong argument: " + arg + "\n")
        sys.exit(1)

if DenseNet:
    from nets.densenet.densenet     import DenseNet169
    ntwk = DenseNet169(UseRelu)
elif ResNet:
    from nets.resnet.resnet         import ResNet50
    ntwk = ResNet50(UseRelu)

##################################################
if True:
    arch = Arch()

    peArray = arch.gPeArray()
    psumBuf = arch.gPsumBuffer()
    pool = arch.gPoolingEng()
    activ = arch.gActivationEng()
    stbuf = arch.gStateBuffer()
    arch.gNumberPsumBanks()
    arch.gPsumBankEntries()
    arch.gNumberPeArrayRows()
    arch.gNumberPeArrayColumns()







##################################################
assert(ntwk)

ntwk.rDoBatching(__DoBatching)
ntwk.construct()
scheduler = Scheduler()
scheduler.Schedule(ntwk)
ntwk.rLevels(scheduler.gLevels())

sbmgr = StateBufferMgr(arch, ntwk)
sbmgr.calcLayerFmapAddresses()

##################################################
printer = Printer(ntwk)

### Printing
if PrintLayers:
    printer.printNetwork()
    print("")

if PrintLevels:
    print("By level")
    printer.printLevels()
    print("")

if PrintSchedule:
    print("By scheduling")
    printer.printSched()
    print("")

if PrintDot:
    print("Dot")
    printer.printDot()
    print("")


