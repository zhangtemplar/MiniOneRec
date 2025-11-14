#!/usr/bin/env python3
import os


if __name__ == "__main__":
    print(os.environ)
    print("WORLD_SIZE=", os.environ.get("WORLD_SIZE"))
    print("RANK=", os.environ.get("RANK"))
    print("LOCAL_RANK=", os.environ.get("LOCAL_RANK"))