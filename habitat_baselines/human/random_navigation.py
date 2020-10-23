#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
from collections import defaultdict
import math
import numpy as np

def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README

    with habitat.Env(
        config=habitat.get_config("habitat_baselines/config/objectnav/random_objectnav_gibson.yaml")
    ) as env:
        print("Environment creation successful")
    
        agg_metrics: Dict = defaultdict(float)
        total_episodes = len(env.episodes) # 1898
        print("total_episodes: %d" % total_episodes)
        count_episodes = 0
        while count_episodes < total_episodes:
            observations = env.reset()

            # print("Agent stepping around inside environment.")
            count_steps = 0
            while not env.episode_over:
                observations = env.step(env.action_space.sample())
                count_steps += 1
                
            # print("Episode finished after {} steps.".format(count_steps))
            metrics = env.get_metrics()
            # if math.isinf(metrics["distance_to_goal"]) or math.isnan(metrics["distance_to_goal"]):
            #     print("error: ", metrics["distance_to_goal"])
            #     print("episode: ", env.current_episode)
            #     break
            # assert (np.any(np.isnan(metrics)) == False), "Reward has nan."

            for m, v in metrics.items():
                agg_metrics[m] += v
            count_episodes += 1

            if count_episodes % 100 == 0:
                print("Episodes Count: %d" % count_episodes, "Total Count: %d" %total_episodes)

                avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
                for k, v in avg_metrics.items():
                    print("{}: {:.3f}".format(k, v))
        
        env.close()
   
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        for k, v in avg_metrics.items():
            print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    example()
