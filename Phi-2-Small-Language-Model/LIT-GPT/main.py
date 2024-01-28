import torch
from utils import *

def main():

    torch.set_float32_matmul_precision("medium")
    setup(
        devices=4,
        train_data_dir=path
    )

if __name__ == "__main__":
    main()

