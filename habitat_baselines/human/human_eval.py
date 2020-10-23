import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
from collections import defaultdict
import random
import argparse

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example(run_type: str):

    config=habitat.get_config("habitat_baselines/config/objectnav/human_objectnav_gibson.yaml")

    config.defrost()
    config.DATASET.SPLIT = run_type
    config.freeze()

    env = habitat.Env(
        config=config
    )
    env.seed(10000)

    observations = env.reset()
    # random.seed(10)
    agg_metrics: Dict = defaultdict(float)

    num_episodes = 20
    count_episodes = 0
    scene_id = env.current_episode.scene_id
    old_scene_id = ''
    category = env.current_episode.goals[0].object_category
    old_category = ''
    while count_episodes < num_episodes:

        observations = env.reset()
        scene_id = env.current_episode.scene_id
        category = env.current_episode.goals[0].object_category

        while scene_id == old_scene_id or category == old_category:
            observations = env.reset()
            observations = env.reset()
            scene_id = env.current_episode.scene_id
            category = env.current_episode.goals[0].object_category

        print("Environment ", count_episodes+1, " creation successful! Total: ", num_episodes)

        print("scene_id", scene_id)
        old_scene_id = scene_id

        print("Destination goal: ", category)
        old_category = category

    
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

        print("Agent stepping around inside environment.")


        count_steps = 0
        while not env.episode_over:
            keystroke = cv2.waitKey(0)

            if keystroke == ord(FORWARD_KEY):
                action = HabitatSimActions.MOVE_FORWARD
                print("action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                action = HabitatSimActions.TURN_LEFT
                print("action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                action = HabitatSimActions.TURN_RIGHT
                print("action: RIGHT")
            elif keystroke == ord(FINISH):
                action = HabitatSimActions.STOP
                print("action: FINISH")
            else:
                print("INVALID KEY")
                continue

            observations = env.step(action)
            count_steps += 1


            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

        print("Episode finished after {} steps.".format(count_steps))

        if (
            action == HabitatSimActions.STOP and 
            env.get_metrics()["spl"]
        ):
            print("you successfully navigated to destination point")
        else:
            print("your navigation was not successful")

        metrics = env.get_metrics()

        # for k, v in metrics.items():
        #     print("{}: {:.3f}".format(k, v))

        for m, v in metrics.items():
            agg_metrics[m] += v
        count_episodes += 1

    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    for k, v in avg_metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "val"],
        required=True,
        help="run type of the experiment (train or eval)",
    )

    args = parser.parse_args()

    example(**vars(args))