import gymnasium
from gymnasium import spaces
import pygame
import numpy as np

FIELD_SIZE = 21
MAX_MOVE = 5
MOVING_REWARD = -2
CREDIT_REWARD = 10
LOSS_REWARD = 0
AVERAGE_REQUEST_0 = 3
AVERAGE_REQUEST_1 = 4
AVERAGE_RETURN_0 = 3
AVERAGE_RETURN_1 = 2
DETERMINED = True


class JackCarRental(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=FIELD_SIZE, max_move = MAX_MOVE):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observation is dictionary with the agent's location.
        # Location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)

        # Current state of Jack's business ([a, b], where a and b are the amount of cars on location A and B)
        self.state = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We have 2*MAX_MOVE + 1 actions, from -MAX_MOVE to MAX_MOVE (The amount of cars we move from A to B)
        self.action_space = spaces.Discrete(max_move*2 + 1, start=-max_move)

        self.accumulated_reward = 0


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return self.accumulated_reward

    def reset(self, seed=None, options=None, state=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if state is None:
            # Choose the agent's location uniformly at random
            self.state = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            self.state = np.array(state, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        initial_state = self.state
        reward = 0

        reward += abs(action) * MOVING_REWARD

        # We use `np.clip` to make sure we don't leave the grid
        self.state = np.clip(self.state + [action, -action], 0, self.size - 1)


        # Generate requests and returns for cars on both locations
        if DETERMINED:
            requests = [AVERAGE_REQUEST_0, AVERAGE_REQUEST_1]
            returns = [AVERAGE_RETURN_0, AVERAGE_RETURN_1]
        else:
            requests = [np.random.poisson(AVERAGE_REQUEST_0), np.random.poisson(AVERAGE_REQUEST_1)]
            returns = [np.random.poisson(AVERAGE_RETURN_0), np.random.poisson(AVERAGE_RETURN_1)]

        # An episode is done if Jack is a bankrupt
        terminated = (self.state[0] < requests[0]) or (self.state[1] < requests[1])

        if not terminated:
            reward += CREDIT_REWARD * (requests[0] + requests[1])
        else:
            reward += LOSS_REWARD + CREDIT_REWARD * (requests[0] + requests[1])

        self.state = np.clip(self.state - requests + returns, 0, self.size - 1)

        if self.render_mode == "human":
            self._render_frame()

        # return observation, reward, terminated, False, info

        return [initial_state, action, self.state, reward], terminated

    def chess_board_transition(self):
        new_state = np.zeros(FIELD_SIZE * FIELD_SIZE)
        new_state[FIELD_SIZE * self.state[0] + self.state[1]] = 1
        return new_state

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.state + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
