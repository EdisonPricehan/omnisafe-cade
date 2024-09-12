
from gymnasium.envs.toy_text.cliffcircular import CliffCircularEnv
import omnisafe


if __name__ == '__main__':
    # env_id = 'CliffWalking-v0'
    # env_id = 'CliffCircular-v0'
    env_id = 'CliffCircular-v1'
    # env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'

    # agent = omnisafe.Agent('PPOLag', env_id)
    # agent = omnisafe.Agent('PPO', env_id)
    # agent = omnisafe.Agent('CCEPETS', env_id)
    # agent = omnisafe.Agent('RCEPETS', env_id)
    # agent = omnisafe.Agent('FOCOPS', env_id)
    agent = omnisafe.Agent('FOCOPS_CACD', env_id)
    # agent = omnisafe.Agent('PETS', env_id)
    # agent = omnisafe.Agent('LOOP', env_id)

    agent.learn()
    # agent.plot(smooth=1)
    # agent.render(num_episodes=1, render_mode='rgb_array', width=128, height=128)
