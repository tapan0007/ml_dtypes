# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Kaena operations (nodes) of uint8 quantization related operations
# Suitable for TF freeze graph input and general op reduction and fusing

import numpy as np
from NpTransforms import NpTrans as npt
from NpTransforms import TensorFormat
from KaenaOpGraph import Node, NodeConv2D


###############################################################################
# Quantization
###############################################################################
class NodeQuantize(Node):
    def genCompilerLayerJson(self, tensorFormatMap):
        fileList = []
        npInfo = self.getNpInfo()[0]
        tpbShape, simFormat, npFileSim = self.convertShape(npInfo, tensorFormatMap)
        tensorFormatMap.add(npInfo.tensorName,
                            TensorFormat(npInfo.tensorName, self.getOpName(),
                                         npInfo.npFile, npt.Formats[npt.TF][npt.Fmaps],
                                         npFileSim, simFormat, False))

        (fromIfNode, npInfoIF), (_, minInputNpInfo), (_, maxInputNpInfo) = self.getInputNodesAndNpInfo()
        minInput = np.asscalar(minInputNpInfo.getValues())
        maxInput = np.asscalar(maxInputNpInfo.getValues())
        quantScale, _, zeroPoint = QuantizeConsts(minInput, maxInput, T=npInfo.dType)
        layerData = {
            "ofmap_shape"       : tpbShape,
            "ofmap_format"      : simFormat,
            "ref_file"          : npFileSim,
            "quant_scale"       : quantScale,
            "zero_point"        : zeroPoint,
            "previous_layers"   : [fromIfNode.getName()],
            "#comment"          : "Quantize %s into %s" % (npInfoIF.dType, npInfo.dType)
        }
        fileList.append(npFileSim)
        layerDataBase, fileListBase = Node.genCompilerLayerJson(self, tensorFormatMap)
        layerDataBase[0].update(layerData)
        fileListBase.extend(fileList)
        return layerDataBase, fileListBase

    def isSupported(self):
        return True


###############################################################################
# Dequantization
###############################################################################
class NodeDequantize(Node):
    def genCompilerLayerJson(self, tensorFormatMap):
        fileList = []
        npInfo = self.getNpInfo()[0]
        tpbShape, simFormat, npFileSim = self.convertShape(npInfo, tensorFormatMap)
        tensorFormatMap.add(npInfo.tensorName,
                            TensorFormat(npInfo.tensorName, self.getOpName(),
                                         npInfo.npFile, npt.Formats[npt.TF][npt.Fmaps],
                                         npFileSim, simFormat, False))
        (fromIfNode, npInfoIF), (_, minInputNpInfo), (_, maxInputNpInfo) = self.getInputNodesAndNpInfo()
        minInput = np.asscalar(minInputNpInfo.getValues())
        maxInput = np.asscalar(maxInputNpInfo.getValues())
        _, dequantScale, zeroPoint = QuantizeConsts(minInput, maxInput, T=npInfoIF.dType)

        layerData = {
            "ofmap_shape"       : tpbShape,
            "ofmap_format"      : simFormat,
            "ref_file"          : npFileSim,
            "dequant_scale"     : dequantScale,
            "zero_point"        : zeroPoint,
            "previous_layers"   : [fromIfNode.getName()],
            "#comment"          : "Dequantize %s into %s" % (npInfoIF.dType, npInfo.dType)
        }
        fileList.append(npFileSim)
        layerDataBase, fileListBase = Node.genCompilerLayerJson(self, tensorFormatMap)
        layerDataBase[0].update(layerData)
        fileListBase.extend(fileList)
        return layerDataBase, fileListBase

    def isSupported(self):
        return True


###############################################################################
# Quantized 2D convolution
###############################################################################
class NodeQuantizedConv2D(NodeConv2D):
    def getInputNodesAndNpInfo(self):
        ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW),
            (_, minInputNpInfo), (_, maxInputNpInfo),
            (_, minFilterNpInfo), (_, maxFilterNpInfo)) = Node.getInputNodesAndNpInfo(self)
        self.quantizeMinInput = np.asscalar(minInputNpInfo.getValues())
        self.quantizeMaxInput = np.asscalar(maxInputNpInfo.getValues())
        self.quantizeMinFilter = np.asscalar(minFilterNpInfo.getValues())
        self.quantizeMaxFilter = np.asscalar(maxFilterNpInfo.getValues())
        return (fromIfNode, npInfoIF), (fromWeightNode, npInfoW)

    def getDataFormat(self):
        return "NHWC"

    # Returns layer json model in dictionary format, and list of files (npy data)
    def genCompilerLayerJson(self, tensorFormatMap):
        fileList = []
        npInfo = self.getNpInfo()[0]
        tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
        (fromIfNode, npInfoIF), (fromWeightNode, npInfoW) = self.getInputNodesAndNpInfo()

        # calculate min_output and max_output
        factorInput = ((self.quantizeMaxInput - self.quantizeMinInput) /
            (np.iinfo(npInfoIF.dType).max - np.iinfo(npInfoIF.dType).min))
        factorFilter = ((self.quantizeMaxFilter - self.quantizeMinFilter) /
            (np.iinfo(npInfoW.dType).max - np.iinfo(npInfoW.dType).min))
        factorOutput = factorInput * factorFilter
        minOutput = (np.iinfo(npInfo.dType).min + 1) * factorOutput # +1 comes from symmetric quantization
        maxOutput = np.iinfo(npInfo.dType).max * factorOutput

        quantScaleInput, dequantScaleInput, zeroPointInput = QuantizeConsts(
            self.quantizeMinInput, self.quantizeMaxInput, T=npInfoIF.dType)
        quantScaleFilter, dequantScaleFilter, zeroPointFilter = QuantizeConsts(
            self.quantizeMinFilter, self.quantizeMaxFilter, T=npInfoW.dType)

        tpbFilterShape = list(npt.reorderShape(npInfoW.npShape, npt.TF, npt.SIM, npt.Weights))
        # OFMAP
        npFileSim, simFormatOF = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
        tensorFormatMap.add(npInfo.tensorName,
                            TensorFormat(npInfo.tensorName, self.getOpName(),
                                         npInfo.npFile, npt.Formats[npt.TF][npt.Fmaps],
                                         npFileSim, simFormatOF, False))
        # IFMAP, not needed
        npFileSimF, simFormatIF  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
        tensorFormatMap.add(npInfoIF.tensorName,
                            TensorFormat(npInfoIF.tensorName, self.getOpName(),
                                         npInfoIF.npFile, npt.Formats[npt.TF][npt.Fmaps],
                                         npFileSimF, simFormatIF, False))
        # WEIGHT
        npFileSimW, simFormatW = npt.copyNpyFileAs(npInfoW.npFile, npt.TF, npt.SIM, npt.Weights)
        tensorFormatMap.add(npInfoW.tensorName,
                            TensorFormat(npInfoW.tensorName, self.getOpName(),
                                         npInfoW.npFile, npt.Formats[npt.TF][npt.Weights],
                                         npFileSimW, simFormatW, True))

        fileList.extend([npFileSimW, npFileSim])
        stride = npt.reorderShape(self.getStrides(), npt.TF, npt.SIM, npt.Fmaps)
        padding = self.calcTpbPadding(self.getFilter2D(), self.getPaddingMode())
        layerData = {
            "layer_type"        : "QuantizedConv",
            "kernel_file"       : npFileSimW,
            "kernel_format"     : simFormatW,
            "kernel_shape"      : tpbFilterShape,
            "dequant_scale_input"   : dequantScaleInput,
            "dequant_scale_filter"  : dequantScaleFilter,
            "zero_point_input"  : zeroPointInput,
            "zero_point_filter" : zeroPointFilter,
            "min_output"        : minOutput,
            "max_output"        : maxOutput,
            "ofmap_shape"       : tpbShape,
            "ofmap_format"      : simFormatOF,
            "ref_file"          : npFileSim,
            "padding"           : padding,
            "previous_layers"   : [fromIfNode.getName()],
            "stride"            : stride,
            "#comment"          : "Quantized two dimensional convolution with explicit padding, "
                                  "%s inputs, %s outputs" % (npInfoIF.dType, npInfo.dType)
            }
        layerDataBase, fileListBase = Node.genCompilerLayerJson(self, tensorFormatMap)
        layerDataBase[0].update(layerData)
        fileListBase.extend(fileList)
        return layerDataBase, fileListBase


"""Helper for finding scale and zeroPoint from minRange and minRange
quantization equation: realValue = scale * (quantizedValue - zeroPoint)
"""
def QuantizeConsts(minRange, maxRange, T):
    # note: T must be an integer type
    maxMinusMin = maxRange - minRange
    intRange = np.iinfo(T).max - np.iinfo(T).min
    quantScale = intRange / maxMinusMin
    dequantScale = maxMinusMin / intRange
    if T in {'uint8', 'uint16', 'uint32', np.uint8, np.uint16, np.uint32}:
        zeroPoint = int(np.round(-minRange * quantScale))
    elif T in {'int8', 'int16', 'int32', np.int8, np.int16, np.int32}:
        zeroPoint = 0
    else:
        raise NotImplementedError('dtype %s is not supported' % T)
    return quantScale, dequantScale, zeroPoint
