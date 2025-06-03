from pynput import keyboard
from typing import Optional, List, Tuple


class Key2ActionDrone:
    """
    A class to convert keyboard inputs into discrete actions for drone waypoint control.
    """
    def __init__(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.last_key = None

    def on_press(self, key):
        try:
            pass
            # print('alphanumeric key {0} pressed'.format(key.char))
        except AttributeError:
            print('special key {0} pressed'.format(key))

    def on_release(self, key):
        if key == keyboard.Key.esc:
            self.close()
            print(f'Keyboard listener is stopped.')
            return

        self.last_key = key

    def close(self):
        self.listener.stop()

    def get_multi_discrete_action(self) -> Optional[List[int]]:
        # Skip human input, let agent action step the environment
        if self.last_key == keyboard.Key.space:
            self.last_key = None
            return None

        # [vertical translation, horizontal rotation, longitudinal translation, latitudinal translation]
        action = [1] * 4
        if self.last_key is None:
            return action

        if self.last_key == keyboard.KeyCode.from_char('w'):
            action[0] = 0
        elif self.last_key == keyboard.KeyCode.from_char('s'):
            action[0] = 2
        elif self.last_key == keyboard.KeyCode.from_char('a'):
            action[1] = 0
        elif self.last_key == keyboard.KeyCode.from_char('d'):
            action[1] = 2
        elif self.last_key == keyboard.KeyCode.from_char('i'):
            action[2] = 0
        elif self.last_key == keyboard.KeyCode.from_char('k'):
            action[2] = 2
        elif self.last_key == keyboard.KeyCode.from_char('j'):
            action[3] = 0
        elif self.last_key == keyboard.KeyCode.from_char('l'):
            action[3] = 2
        else:
            print(f'Unrecognized key {self.last_key}')

        self.last_key = None
        return action


class Key2ActionBoat:
    """
    A class to convert keyboard inputs into continuous actions for boat thruster control.
    """
    def __init__(self, step_size: float = 0.1):
        self.step_size: float = step_size
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.last_key = None

    def on_press(self, key):
        try:
            pass
            # print('alphanumeric key {0} pressed'.format(key.char))
        except AttributeError:
            print('special key {0} pressed'.format(key))

    def on_release(self, key):
        if key == keyboard.Key.esc:
            self.close()
            print(f'Keyboard listener is stopped.')
            return

        self.last_key = key

    def close(self):
        self.listener.stop()

    def get_continuous_action(self) -> Optional[Tuple[float, float]]:
        # Skip human input, let agent action step the environment
        if self.last_key == keyboard.Key.space:
            self.last_key = None
            return None

        default_action: Tuple[float, float] = 0., 0.
        if self.last_key is None:
            return default_action

        action: Optional[Tuple[float, float]] = None
        if self.last_key == keyboard.KeyCode.from_char('w'):
            action = self.step_size, 0.
        elif self.last_key == keyboard.KeyCode.from_char('s'):
            action = -self.step_size, 0.
        elif self.last_key == keyboard.KeyCode.from_char('i'):
            action = 0., self.step_size
        elif self.last_key == keyboard.KeyCode.from_char('k'):
            action = 0., -self.step_size
        else:
            print(f'Unrecognized key {self.last_key}')

        self.last_key = None
        return default_action if action is None else action

