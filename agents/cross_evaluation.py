import asyncio

from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from tabulate import tabulate


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
    # We create three random players
    players = [
        RandomPlayer(max_concurrent_battles=10),
        # MaxDamagePlayer(max_concurrent_battles=10),
        RLPlayer(max_concurrent_battles=10),
        RLPlayer_high_vic(max_concurrent_battles=10),
        RLPlayer_high_hp(max_concurrent_battles=10),
        RLPlayer_high_faint(max_concurrent_battles=10),
        RLPlayer_diff_model(max_concurrent_battles=10)
        ]

    # Now, we can cross evaluate them: every player will player 20 games against every
    # other player.
    cross_evaluation = await cross_evaluate(players, n_challenges=20)

    # Defines a header for displaying results
    table = [["-"] + [p.username for p in players]]

    # Adds one line per player with corresponding results
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Displays results in a nicely formatted table.
    print(tabulate(table))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())