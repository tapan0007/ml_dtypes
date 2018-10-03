import os, sys
import numpy as np
import argparse

sys.path.insert(0, os.environ["KAENA_PATH"] + "/compiler")

from tffe.NpTransforms import NpTrans

def ConvertBFP16FP32(arr):
  arr_bytes = bytearray(arr.tobytes())
  assert len(arr_bytes) % 2 == 0, "Input must be multiple of 2"

  new_bytes = bytearray()

  # Check the input data type so we know which way to convert
  if arr.dtype == '|V2' or arr.dtype == np.float16:
    print("Converting from bfloat16 to float32")
    result_dtype = np.float32
    for i in range(0, len(arr_bytes)//2):
      new_bytes.extend([0, 0, arr_bytes[2*i], arr_bytes[2*i+1]])
  elif arr.dtype == np.float32:
    print("Converting from float32 to bfloat16 (stored in float16 container)")
    result_dtype = np.float16  # store bfloat16 in float16 container
    for i in range(0, len(arr_bytes)//4):
      new_bytes.extend([arr_bytes[4*i+2], arr_bytes[4*i+3]])
  else:
    assert 0, "Invalid input type"

  return np.frombuffer(new_bytes, dtype=result_dtype).reshape(arr.shape)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", help="Input file", required=True)
  parser.add_argument("--output", help="Output file", required=True)
  parser.add_argument("--toNCHW", help="Convert the input from NHWC to NCHW", action="store_true")
  parser.add_argument("--toNHWC", help="Convert the input from NCHW to NHWC", action="store_true")
  parser.add_argument("--toNC", help="Convert the input from (N,C,1,1) to (N,C)", action="store_true")
  args = parser.parse_args()

  inarray = np.load(args.input)
  outarray = ConvertBFP16FP32(inarray)
  if args.toNCHW:
    print("Transposing from NHWC to NCHW...")
    outarray = NpTrans.formatNpyArrAs(outarray, NpTrans.NHWC, NpTrans.NCHW)
  elif args.toNHWC:
    print("Transposing from NCHW to NHWC...")
    outarray = NpTrans.formatNpyArrAs(outarray, NpTrans.NCHW, NpTrans.NHWC)
  elif args.toNC:
    assert outarray.shape[2] == outarray.shape[3] == 1, "Can't convert to NC format, need (N,C,1,1) dims"
    print("Removing extra dimensions to reduce to NC...")
    outarray = outarray[:,:,0,0]
  np.save(args.output, outarray)

