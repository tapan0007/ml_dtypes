#!/usr/bin/env python3

# Copyright (C) 2017, Amazon.com. All Rights Reserved

import sys, re

# Usage
#  # use  tf_conv2d_extract_padding.py  first to create log-gpu, then run:
#  grep S= log-gpu |  gold_validate_padding_calc.py | grep False

def calcPad(S, R, H):
  Ho = (H + S - 1) // S
  spacing = S - R  # negative spacing means overlap
  inPixels = Ho * R + (Ho - 1) * spacing
  L = max(0, inPixels - H)
  return(True, L // 2, (L + 1) // 2)

for line in sys.stdin:
#for line in ['S=2 R=3 H=3 PadLeft= 1 PadRight= 1\n']:
  print(line, end='')
  (S, R, H, Pl, Pr) = [int(x) for x in re.findall('\d+', line)]
  #print(S, R, H, Pl, Pr)
  (isSupported, calcPl, calcPr) = calcPad(S, R, H)
  ok = (calcPl == Pl) and (calcPr == Pr)
  print("    S=%d R=%d H=%d    isSupported=%-5s  Pl(calc)=%2d(%2d)  Pr(calc)=%2d(%2d)  ok=%s" %
        (S, R, H, isSupported, Pl, calcPl, Pr, calcPr, ok))
  print()
