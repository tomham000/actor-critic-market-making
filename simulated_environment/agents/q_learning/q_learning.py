from typing import Tuple, List, Dict
import random


class QLearningAgent:
    def __init__(self, actions: List[Tuple[float, float]], alpha: float, gamma: float):
        self.q_table: Dict[Tuple[Tuple[float, int, int], Tuple[float, float]], float] = {}
        self.epsilon = 0.15
        self.alpha = alpha
        self.gamma = gamma
        self.actions: List[Tuple[float, float]] = actions

    def get_q_value_for_state(self, state, action) -> float:
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state: Tuple[float, int, int], ignore_exploration=False) -> Tuple[float, float]:
        q = [self.get_q_value_for_state(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon and not ignore_exploration:
            minQ = min(q);
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        return action

    def learn(self, state1: Tuple[float, int, int], action: Tuple[float, float], reward: float,
              state2: Tuple[float, int, int]):
        max_q_new_state = max([self.get_q_value_for_state(state2, a) for a in self.actions])
        old_q_value = self.get_q_value_for_state(state1, action)
        self.q_table[(state1, action)] = old_q_value + self.alpha * (
                    reward + self.gamma * max_q_new_state - old_q_value)
