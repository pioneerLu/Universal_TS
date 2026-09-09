#!/usr/bin/env python3
"""Generate small synthetic series."""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t = np.arange(2048, dtype=np.float32)
    uni = (np.sin(t / 24.0) + 0.1 * rng.standard_normal(t.shape)).astype(np.float32)
    uni = uni.reshape(1, -1, 1)
    np.save(os.path.join(args.out, "univariate.npy"), uni)

    wind = np.sin(t / 18.0) + 0.05 * rng.standard_normal(t.shape)
    power = 0.6 * np.clip(wind, 0, None) ** 3 + 0.05 * rng.standard_normal(t.shape)
    humid = 0.5 + 0.1 * np.sin(t / 48.0)
    multi = np.stack([wind, humid, power], axis=1).astype(np.float32)
    np.save(os.path.join(args.out, "wind_power.npy"), multi)
    print(f"wrote {args.out}/univariate.npy {uni.shape}")
    print(f"wrote {args.out}/wind_power.npy {multi.shape}")


if __name__ == "__main__":
    main()
