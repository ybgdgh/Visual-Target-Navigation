import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp

import tqdm

import habitat
import habitat_sim
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode

num_episodes_per_scene = int(1e2)


def _generate_fn(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = []
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    print(vars(dset))
    dset.episodes = list(
        generate_pointnav_episode(
            sim, num_episodes_per_scene, is_gen_shortest_path=False
        )
    )
    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]

    scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
    out_file = f"./data/datasets/pointnav/gibson/v1/all/content/{scene_key}.json.gz"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())


scenes = glob.glob("./data/scene_datasets/gibson/*.glb")
with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
    for _ in pool.imap_unordered(_generate_fn, scenes):
        pbar.update()

with gzip.open(f"./data/datasets/pointnav/gibson/v1/all/all.json.gz", "wt") as f:
    json.dump(dict(episodes=[]), f)