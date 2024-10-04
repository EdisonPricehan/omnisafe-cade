
import omnisafe
from omnisafe.envs.riverine_env import RiverineEnv


if __name__ == '__main__':
    env_id = 'Medium'

    agent = omnisafe.Agent('FOCOPS_CACD', env_id)

    agent.learn()


