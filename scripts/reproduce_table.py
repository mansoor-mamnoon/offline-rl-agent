#!/usr/bin/env python
# TODO: reproduce benchmark table
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="traffic")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--n-epochs", type=int, default=10)
    args = parser.parse_args()
    print(f"Reproducing table for env={args.env}, seeds={args.seeds}, n_epochs={args.n_epochs}")
    print("TODO: implement full reproduction")
