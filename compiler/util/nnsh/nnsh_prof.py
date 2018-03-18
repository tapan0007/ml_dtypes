# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Neural network scheduler - profile of execution on TPB
#

import csv
import nnsh_cost

class NnshProfile(object):
  
  def __init__(self):
    self.steps = []

  def addStep(self, stepCosts):
    self.steps.append(stepCosts)
  
  def writeCsv(self, csvFile):
    
    with open(csvFile, 'w') as csvHandle:
      fieldNames = ["StepId", "WeightsMiB", "IfmapsMiB", "OfmapsMiB",
                    "Notes"]
      rows = []
      for s in range(len(self.steps)):
        row = {}
        row["StepId"] = s
        row["WeightsMiB"] = 0
        row["IfmapsMiB"] = 0
        row["OfmapsMiB"] = 0
        row["Notes"] = ""
        for c in self.steps[s]:
          row["WeightsMiB"] += c.bytesW / 2 ** 20
          row["IfmapsMiB"]  += c.bytesIf / 2 ** 20
          row["OfmapsMiB"]  += c.bytesOf / 2 ** 20
          row["Notes"] += "    " + str(c)
        rows.append(row)
      writer = csv.DictWriter(csvHandle, fieldnames=fieldNames)
      writer.writeheader()
      writer.writerows(rows)
    print("INFO: Wrote op sequences into " + csvFile)
  
  
  
  
