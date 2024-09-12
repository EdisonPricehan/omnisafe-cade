# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""One example for evaluate saved policy."""

import os

import torch

import omnisafe

from omnisafe.envs import DiscreteEnv
from gymnasium.envs.toy_text.cliffcircular import CliffCircularEnv


# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
# LOG_DIR = './runs/PPO-{CliffCircular-v0}/seed-002-2024-02-18-14-36-59'
# LOG_DIR = './runs/PETS-{CliffCircular-v0}/seed-000-2024-03-10-16-34-45'
# LOG_DIR = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-08-26-16-29-20'
# LOG_DIR = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-09-09-23-51-15'
# LOG_DIR = './runs/FOCOPS-{CliffCircular-v1}/seed-000-2024-09-10-14-28-38'
LOG_DIR = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-09-12-14-22-35'


if __name__ == '__main__':
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    scan_dir = os.scandir(os.path.join(LOG_DIR, 'torch_save'))
    for item in scan_dir:
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            if '600' not in item.name:
                continue
            else:
                print(f'Start evaluating {item.name} ...')

            # model = torch.load(os.path.join(LOG_DIR, 'torch_save', item.name))
            # print(model.keys())
            # print(type(model['actor_critic']))
            # print(model['actor_critic'].keys())

            evaluator.load_saved(
                save_dir=LOG_DIR,
                model_name=item.name,
                camera_name='track',
                width=5,
                height=5,
            )
            # evaluator.render(num_episodes=1)
            evaluator.evaluate(num_episodes=20)
    scan_dir.close()
