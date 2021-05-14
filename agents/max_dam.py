import time
import gym
import numpy as np
import tensorflow as tf
import random
import rl
import os

from tabulate import tabulate

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from collections import deque

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
from poke_env.player.utils import cross_evaluate


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


async def main():
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(
        battle_format="gen8randombattle",
    )
    max_damage_player = MaxDamagePlayer(
        battle_format="gen8randombattle",
    )

    # Now, let's evaluate our player
    #await max_damage_player.send_challenges("RLPokeBot", n_challenges=1)

    # print(
    #     "Max damage player won %d / 100 battles [this took %f seconds]"
    #     % (
    #         max_damage_player.n_won_battles, time.time() - start
    #     )
    # )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())