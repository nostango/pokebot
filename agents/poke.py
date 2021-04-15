import asyncio
import time
import gym
import numpy as np
import random

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration


class RLPlayer(Player):
    ### this will contain the necessary observations of a battle needed for the learning:
    ### number of pokemon (opp & player), move base power, move type multi, move category (phyisical, )
    def battle_components(self, battle):

        ### the vector containing all the information needed
        result = np.array([])

        ### gets the number of pokemon in opponent's and player's party and adds them to the final vector
        opp_remaining_pokemon = battle.opponent_active_pokemon
        player_remaining_pokemon = battle.active_pokemon

        result.append(opp_remaining_pokemon)
        result.append(player_remaining_pokemon)

        ### retrieves the base power and the type multiplier of each move my pokemon has
        moves_base_power = np.array([0, 0, 0, 0])
        move_type_multiplier = np.array([1, 1, 1, 1])
        move_category = np.array([0, 0, 0, 0])
        for i, move in ennumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type != None:
                move_type_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2
                )
            move_category.append(move.move_category)

        ### add the results to the final vector
        result.append(moves_base_power)
        result.append(move_type_multiplier)
        result.append(move_category)

        return result

    def reward(self, battle):
        compute_reward(
            battle,
            fainted_value = 3,
            hp_value = 1,
            status_value = 0.5,
            victory_value = 50
        )

# REPLAY_MEMORY_SIZE = 10_000
# MIN_REPLAY_MEMORY_SIZE = 1_000

# class DQNAgent:
#         def __init__(self):

#         self.model = self.create_model()

#         self.target_model = self.create_model()
#         self.target_model.set_weights(self.model.get_weights())

#         # An array with last n steps for training
#         self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

#         # Used to count when to update target network with main network's weights
#         self.target_update_counter = 0



#     def create_model(self):
#         n_action = len(env_player.action_space)
#         model = Sequential()
#         model.add(Conv2D(256, input_shape=(1, 13))
#         model.add(Activation("relu"))
#         model.add(MaxPooling2D(2, 2))
#         model.add(Dropout(0.2))

#         model.add(Conv2D(256), (1, 3)
#         model.add(Activation("relu"))
#         model.add(MaxPooling2D(2, 2))
#         model.add(Dropout(0.2))

#         model.add(Flatten())
#         model.add(Dense(64))

#         model.add(Dense(n_action, activation="linear"))
#         model.compile(Loss="mse", optimizer=Adam(lr=0.0025), metrics=['mae'])
#         return model
    
#     ### transition is observation space
#     def update_replay_memory(self, transition):
#         self.replay_memory.append(transition)

#     def get_qs(self, state):
#         return self.model_predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
if __name__ == "__main__":
    ### get the players
    env_player = RLPlayer(battle_format="gen8randombattle")
    opponent = RandomPlayer(battle_format="gen8randombattle")

    ### create the model
    n_action = len(env_player.action_space)

    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))

    # Our embedding have shape (1, 10), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
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
