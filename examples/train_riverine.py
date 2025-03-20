
import omnisafe
from omnisafe.envs.riverine_env import RiverineEnv


if __name__ == '__main__':
    env_id = 'medium'

    agent = omnisafe.Agent('FOCOPS_CADE', env_id)

    agent.learn()


