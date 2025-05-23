from __future__ import annotations
import math
from copy import deepcopy
import time
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose
import random


# You only need to modify the TreeNode!
class TreeNode:
    # You can change this to include other attributes. 
    # param is the value passed via the -p command line option (default: 0.5)
    # You can use this for e.g. the "c" value in the UCB-1 formula
    def __init__(self, param, parent=None):
        self.children = {}
        self.parent = parent
        self.results = []
        self.param = param
        self.sc = 0.5
        self.action = None
        self.aname = None
    
    # REQUIRED function
    # Called once per iteration
    def step(self, state):
        self.select(state)

        
    # REQUIRED function
    # Called after all iterations are done; should return the 
    # best action from among state.get_actions()
    def get_best(self, state):

        amax = None
        cmax_weight = -1
        for a in state.get_actions():
            if a.key() in self.children:
                c = self.children[a.key()]
                cweight = c.sc
                if cweight > cmax_weight:
                    amax = a
                    cmax_weight = cweight
        if amax is not None:
            return amax
        return random.choice(state.get_actions())
        
    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent = 0):
        for c in self.children.values():
            s = "\t" * indent
            s += c.aname + ": " + str(int(c.sc * 100) / 100) + " " + str(len(c.results))
            print(s)
            c.print_tree(indent + 1)


    # RECOMMENDED: select gets all actions available in the state it is passed
    # If there are any child nodes missing (i.e. there are actions that have not 
    # been explored yet), call expand with the available options
    # Otherwise, pick a child node according to your selection criterion (e.g. UCB-1)
    # apply its action to the state and recursively call select on that child node.
    def select(self, state):
        if state.ended():
            self.backpropagate(self.score(state))
            return
        statecopy = state.copy_undeterministic()
        available_actions = []
        actions = statecopy.get_actions()
        for a in statecopy.get_actions():
            if a.key() not in self.children:
                available_actions.append(a)
        if len(available_actions) > 0:
            self.expand(statecopy, available_actions)
            return
        amax = None
        cmax = None
        cmax_weight = -1
        for a in statecopy.get_actions():
            c = self.children[a.key()]
            cweight = c.sc + self.param * math.sqrt(math.log(len(self.results) / len(c.results)))
            if cweight > cmax_weight:
                amax = a
                cmax = c
                cmax_weight = cweight
        statecopy.step(amax)
        cmax.select(statecopy)


    # RECOMMENDED: expand takes the available actions, and picks one at random,
    # adds a child node corresponding to that action, applies the action ot the state
    # and then calls rollout on that new node
    def expand(self, state, available):
        statecopy = state.copy_undeterministic()
        choice = random.choice(available)
        nn = TreeNode(self.param, self)
        self.children[choice.key()] = nn
        nn.action = choice.key()
        nn.aname = str(choice)
        statecopy.step(choice)
        nn.rollout(statecopy)

    # RECOMMENDED: rollout plays the game randomly until its conclusion, and then 
    # calls backpropagate with the result you get 
    def rollout(self, state):
        while not state.ended():
            action = random.choice(state.get_actions())
            state.step(action)
        self.backpropagate(self.score(state))
        
    # RECOMMENDED: backpropagate records the score you got in the current node, and 
    # then recursively calls the parent's backpropagate as well.
    # If you record scores in a list, you can use sum(self.results)/len(self.results)
    # to get an average.
    def backpropagate(self, result):
        self.results.append(result)
        self.sc = sum(self.results)/len(self.results)
        if self.parent is not None:
            self.parent.backpropagate(result)
        
    # RECOMMENDED: You can start by just using state.score() as the actual value you are 
    # optimizing; for the challenge scenario, in particular, you may want to experiment
    # with other options (e.g. squaring the score, or incorporating state.health(), etc.)
    def score(self, state):
        if state.ended():
            if state.get_end_result() == 1:
                return 1
            # return 0
        return (state.score() + state.health()) / 3
        
        
# You do not have to modify the MCTS Agent (but you can)
class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    # REQUIRED METHOD
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            return actions[0].to_action(battle_state)
    
        t = TreeNode(self.param)
        start_time = time.time()

        for i in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)
        
        best_action = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()
        
        if best_action is None:
            print("WARNING: MCTS did not return any action")
            return random.choice(self.get_choose_card_options(game_state, battle_state)) # fallback option
        return best_action.to_action(battle_state)
    
    # REQUIRED METHOD: All our scenarios only have one enemy
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    # REQUIRED METHOD: Our scenarios do not involve targeting cards
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]
