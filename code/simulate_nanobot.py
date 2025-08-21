#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def simulate(grid_size=50, steps=200, out_img="images/nanobot_path.png", out_csv="outputs/nanobot_traj.csv"):
    np.random.seed(1)
    start = np.array([2.0, 2.0])
    target = np.array([grid_size-10.0, grid_size-10.0])

    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    dist = np.sqrt((xx-target[0])**2 + (yy-target[1])**2)
    pH = 7.4 - (dist / dist.max()) * 2.0

    pos = start.copy()
    traj = [pos.copy()]

    gx, gy = np.gradient(pH)
    for _ in range(steps):
        step = -np.array([gx[int(pos[1]), int(pos[0])], gy[int(pos[1]), int(pos[0])]])
        nrm = np.linalg.norm(step)+1e-8
        pos += step / nrm
        pos = np.clip(pos, 0, grid_size-1)
        traj.append(pos.copy())
        if np.linalg.norm(pos - target) < 1.5:
            break

    traj = np.array(traj)
    Path(Path(out_img).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(out_csv).parent).mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(pH, origin='lower')
    plt.plot(traj[:,0], traj[:,1])
    plt.scatter([start[0], target[0]],[start[1], target[1]])
    plt.title("Nanobot gradient-following (payload near low pH)")
    plt.savefig(out_img, dpi=200, bbox_inches='tight')

    np.savetxt(out_csv, traj, delimiter=",", header="x,y", comments="")
    print("Saved:", out_img, "and", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanobot delivery simulation on pH gradient")
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out_img", type=str, default="images/nanobot_path.png")
    parser.add_argument("--out_csv", type=str, default="outputs/nanobot_traj.csv")
    args = parser.parse_args()
    simulate(grid_size=args.grid_size, steps=args.steps, out_img=args.out_img, out_csv=args.out_csv)
