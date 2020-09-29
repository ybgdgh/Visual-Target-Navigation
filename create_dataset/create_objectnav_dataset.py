import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp

import tqdm

import habitat
import habitat_sim
from habitat.datasets.object_nav.objectnav_generator import generate_objectnav_episode, generate_objectnav_goals_by_category, generate_objectnav_task_category_id

import matplotlib.pyplot as plt
import numpy as np

num_episodes_per_train_scene = int(1e2)
num_episodes_per_val_scene = int(1e1)

task_category = {
    "chair": 1,
    "potted plant": 2,
    "sink": 3,
    "vase": 4,
    "book": 5,
    "couch": 6,
    "bed": 7,
    "bottle": 8,
    "dining table": 9,
    "toilet": 10,
    "refrigerator": 11,
    "tv": 12,
    "clock": 13,
    "oven": 14,
    "bowl": 15,
    "cup": 16,
    "bench": 17,
    "microwave": 18,
    "suitcase": 19,
    "umbrella": 20,
    "teddy bear": 21
}

task_category_dataset_static = dict()

count_scene = 0
def _generate_fn(scene):
    global task_category_dataset_static
    global count_scene
    # print("episode start")
    cfg = habitat.get_config(config_paths="configs/tasks/objectnav_gibson.yaml")
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = []
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    # print(vars(sim))
    # print(sim.config.SCENE)
    sceness = sim.semantic_annotations()
    # print(len(sceness.objects))
    if(len(sceness.objects) == 0): return

    ######################################################################
    ## test semantic annotations
    ######################################################################
    # for obj in sceness.objects:
    #     obj.aabb
    #     obj.aabb.sizes
    #     obj.aabb.center
    #     obj.id
    #     obj.obb.rotation
    #     obj.category.name()
    #     obj.category.index()

    # for region in scene.regions:
    #     region.id
    #     region.category.name()
    #     region.category.index()

    # for level in scene.levels:
    #     level.id

    # print("object num :", len(sceness))
    print("region num :", len(sceness.regions))
    # print("room num :",len(sceness.rooms))
    

    ######################################################################
    ## 统计数据集中的类别
    ######################################################################
    # for obj in sceness.objects:
    #     if obj is not None:
    #         if obj.category.name() in task_category_dataset_static.keys():
    #             task_category_dataset_static[obj.category.name()] += 1
    #         else:
    #             task_category_dataset_static[obj.category.name()] = 1

            # print(
            #     f"Object id:{obj.id}, category:{obj.category.name()}, index:{obj.category.index()}"
            #     f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            #     f" parent_room:{obj.region}"
            # )

    # print(len(sceness.regions))

    # print(task_category_dataset_static)
    # plt.tick_params(axis='x', labelsize=8)    # 设置x轴标签大小
    # plt.tick_params(axis='y', labelsize=4)    # 设置x轴标签大小
    # g = sorted(task_category_dataset_static.items(), key=lambda item:item[1])
    # # print(g)
    # # print(type(g[0][1]), g[0][1])
    # blist=[]
    # clist=[]
    # for key in g:
    #     blist.append(key[0])
    #     clist.append(key[1])
    # plt.barh(range(len(blist)), clist, tick_label=blist, align="center", color="c")
    # #添加图形属性
    # plt.ylabel('Category')
    # plt.xlabel('Number')
    # plt.title('Gibson Dataset Category Statistics')
    # plt.grid()
    # plt.savefig("Gibson_statisitc.jpg", dpi=600)
    #######################################################################


    scene_key = osp.basename(scene)[:-4]
    # print(scene_key)
    if count_scene % 10 == 0:
        num_episodes_per_scene = num_episodes_per_val_scene
        out_file = f"./data/datasets/objectnav/gibson/v1/val/content/{scene_key}.json.gz"
    else:
        out_file = f"./data/datasets/objectnav/gibson/v1/train/content/{scene_key}.json.gz"
        num_episodes_per_scene = num_episodes_per_train_scene
        

    dset = habitat.datasets.make_dataset("ObjectNav-v1")
    # print(vars(dset))
    # print(num_episodes_per_scene)
    dset.goals_by_category = dict(
        generate_objectnav_goals_by_category(
            sim,
            task_category
        )
    )
    # print(len(dset.goals_by_category))

    
    dset.episodes = list(
        generate_objectnav_episode(
            sim, task_category, num_episodes_per_scene, is_gen_shortest_path=True
        )
    )
    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]

    # dset.category_to_task_category_id = generate_objectnav_task_category_id(sim, task_category)
    dset.category_to_task_category_id = task_category
    # dset.category_to_task_category_id = task_category
    dset.category_to_scene_annotation_category_id = dset.category_to_task_category_id

    
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
    count_scene=count_scene+1
    # print(count_scene)
    # print("episode finish!")


scenes = glob.glob("./data/scene_datasets/gibson/*.glb")
# scenes = glob.glob("./data/scene_datasets/mp3d/*/*.glb")
# scenes = glob.glob("./data/scene_datasets/gibson/Emmaus.glb")
print(scenes)
print(len(scenes))
with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
    for _ in pool.imap_unordered(_generate_fn, scenes):
        pbar.update()

with gzip.open(f"./data/datasets/objectnav/gibson/v1/train/train.json.gz", "wt") as f:
    json.dump(dict(episodes=[], 
        category_to_task_category_id=task_category, 
        category_to_scene_annotation_category_id=task_category), f)

with gzip.open(f"./data/datasets/objectnav/gibson/v1/val/val.json.gz", "wt") as f:
    json.dump(dict(episodes=[], 
        category_to_task_category_id=task_category, 
        category_to_scene_annotation_category_id=task_category), f)

