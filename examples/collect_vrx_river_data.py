import os
import csv

import numpy as np
from loguru import logger

from vrx_gym.river_follow_env import WamvGazeboEnv

from omnisafe.utils.key2action import Key2ActionBoat
from omnisafe.utils.patchification import inflate_patch_mask, get_patchified_mask


class VrxDemoCollector:
    def __init__(
        self,
        save_path: str,
        episodes: int,
    ):
        # Init variables
        self._save_path: str = save_path
        self._episodes: int = episodes

        # Init vrx gym env
        self.env = WamvGazeboEnv(
            img_height=128,
            img_width=128,
            incremental_action=True,
            episode_max_step=2000,
            obs_timeout_sec=2.,
            render_mode='human'
        )

        # Init keyboard reader
        self.k2a = Key2ActionBoat(step_size=0.1)

    def run(self):
        cur_episode: int = 0

        try:
            while cur_episode < self._episodes:
                (img, mask), info = self.env.reset()
                done = False

                while not done:
                    thrust_delta_left, thrust_delta_right = 0., 0.

                    # Get human action
                    action = self.k2a.get_continuous_action()
                    if action is not None:
                        thrust_delta_left, thrust_delta_right = action


                    # Step env with human action
                    (img, mask), reward, terminated, truncated, info = self.env.step(action=np.array([thrust_delta_left, thrust_delta_right]))

                    cur_act = self.env.current_action
                    logger.info(f'Delta thrust left: {thrust_delta_left:.1f}, right: {thrust_delta_right:.1f}, thrust left: {cur_act[0]}, thrust right: {cur_act[1]}')

                    # Post-process obs
                    patchified_mask = get_patchified_mask(
                        mask=mask,
                        is_uint8=True,
                    )
                    # logger.info(f'{patchified_mask=}')

                    # Save to file
                    # TODO: add code later

                    # Check episode done
                    done = terminated or truncated
                    if done:
                        cur_episode += 1

        except KeyboardInterrupt:
            self.close()
            logger.error(f'Process is interrupted by keyboard.')

    def close(self) -> None:
        self.env.close()
        self.k2a.close()


if __name__ == '__main__':
    save_path: str = './evaluations/vrx'
    eval_episodes: int = 5

    collector = VrxDemoCollector(save_path=save_path, episodes=eval_episodes)
    collector.run()
