import asyncio
import time
import gym
import numpy as np
import tensorflow as tf
import random
import rl
import os

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.compat.v1.keras.callbacks import TensorBoard
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session
from collections import deque

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration


class RLPlayer(Gen8EnvSinglePlayer):
    ### this will contain the necessary observations of a battle needed for the learning:
    ### number of pokemon (opp & player), move base power, move type multi, move category (phyisical, )
    def embed_battle(self, battle):

        ### gets the number of pokemon in opponent's and player's party and adds them to the final vector
        opp_remaining_pokemon = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        player_remaining_pokemon = len([mon for mon in battle.team.values() if mon.fainted]) / 6


        ### retrieves the base power and the type multiplier of each move my pokemon has
        moves_base_power = np.array([0, 0, 0, 0])
        move_type_multiplier = np.array([1, 1, 1, 1])
        #move_category = np.array([])
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type != None:
                move_type_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2
                )
            #np.append(move_category, move_category.MoveCategory)


        return np.concatenate([
            moves_base_power,
            move_type_multiplier,
            [opp_remaining_pokemon,
            player_remaining_pokemon]
            #move_category,
        ])

    def reward(self, battle):
        compute_reward(
            battle,
            fainted_value = 3,
            hp_value = 1,
            status_value = 0.5,
            victory_value = 50
        )

NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

tf.random.set_seed(0)
np.random.seed(0)

tf_config = os.environ.get('TF_CONFIG')
sess = tf.compat.v1.Session(config=tf_config)

graph = tf.compat.v1.get_default_graph()

def dqn_training(player, dqn, nb_steps):
    global graph
    global sess
    with sess.as_default():
        with sess.graph.as_default():
            set_session(sess)
            dqn.fit(player, nb_steps=nb_steps)
            player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )
    
if __name__ == "__main__":
    ### get the players
    env_player = RLPlayer(battle_format="gen8randombattle")
    opponent = RandomPlayer(battle_format="gen8randombattle")

    ### create the model
    n_action = len(env_player.action_space)

    set_session(sess)
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(1, 10)))

    # Our embedding have shape (1, 10), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(n_action, activation="linear"))

    # model = Sequential()
    # model.add(Conv2D(256, input_shape=(1, 13), activation="relu") ### 13 because of our 13 inputs from the environment

    # model.add(MaxPooling2D(2, 2))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(256), (1, 3), activation="relu")
    # model.add(MaxPooling2D(2, 2))
    # model.add(Dropout(0.2))

    # model.add(Flatten())
    # model.add(Dense(64))

    # model.add(Dense(n_action, activation="linear"))
    # model.compile(Loss="mse", optimizer=Adam(lr=0.0025), metrics=['mae'])

    memory = SequentialMemory(limit=10000, window_length=1)

    # Ssimple epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )

    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )

    dqn.compile(Adam(lr=0.00025), metrics=["mae"])

    set_session(sess)
    
    # Training
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS},
    )
    model.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )
