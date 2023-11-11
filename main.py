import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(DEVICE)

if __name__ == "__main__":
    main()