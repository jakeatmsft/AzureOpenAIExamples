import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os

parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--model_output", type=str, help="Path of output model", default="" )

args = parser.parse_args()

print("hello training world...")

lines = [
    f"Training data path: {args.training_data}",
]

for line in lines:
    print(line)

print("mounted_path files: ")
arr = os.listdir(args.training_data)
print(arr)

for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.training_data, filename), "r") as handle:
        print(handle.read())




