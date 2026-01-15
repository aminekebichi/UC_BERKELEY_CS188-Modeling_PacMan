# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    # available methods under SearchProblem class -->
    # - getStartState(self): Returns the start state for the search problem.
    # - isGoalState(self, state): Returns true if state is a valid goal state
    # - getSuccessors(self, state): For a given state, return list (successor, action, stepCost)
    # - getCostOfActions(self, actions): returns the total cost of a particular sequence of actions.

    frontier = util.Stack() # store tuples of (state, path)
    visited = []
    frontier.push((problem.getStartState(), []))

    while not frontier.isEmpty():
        curr_state, curr_path = frontier.pop()                  # pop the most recent node from the stack (LIFO)

        if curr_state in visited: continue                      # skip any previously visited nodes to avoid cycles
        if problem.isGoalState(curr_state): return curr_path    # return path if goal found

        visited.append(curr_state)
        # process successors
        for state, action, cost in problem.getSuccessors(curr_state):
            if state not in visited:
                    # add to frontier for future exploration
                    frontier.push((state, curr_path + [action]))

    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    frontier = util.Queue() # store tuples of (state, path)
    visited = []
    frontier.push((problem.getStartState(), []))

    while not frontier.isEmpty():
        curr_state, curr_path = frontier.pop()                  # pop the most recent node from the queue (FIFO)

        if curr_state in visited: continue                      # skip any previously visited nodes to avoid cycles
        if problem.isGoalState(curr_state): return curr_path    # return path if goal found

        visited.append(curr_state)

        # process successors
        for state, action, cost in problem.getSuccessors(curr_state):
            if state not in visited:
                    # add to frontier for future exploration
                    frontier.push((state, curr_path + [action]))
    
    return []


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    frontier = util.PriorityQueue()     # store tuples of ((state, path, cost), priority))
    best_costs = {}                     # map states to their best path's cost (lowest cost)

    frontier.push((problem.getStartState(), [], 0), 0)
    best_costs[problem.getStartState()] = 0

    while not frontier.isEmpty():
        curr_state, curr_path, path_cost = frontier.pop()                   # pop the most recent node from the queue (FIFO)
        
        if problem.isGoalState(curr_state): return curr_path                # return path if goal found

        if best_costs.get(curr_state, float('inf')) < path_cost: continue   # if curr_state already has more optimal path, skip
                                                                            # float('inf') used to handle non-existing states
        # process successors
        for state, action, cost in problem.getSuccessors(curr_state):
            new_cost = path_cost + cost
            new_path = curr_path + [action]
    
            if state not in best_costs or new_cost < best_costs[state]:
                best_costs[state] = new_cost  # Update the best known cost
                # Either push new node or update existing one
                frontier.update((state, new_path, new_cost), new_cost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    frontier = util.PriorityQueue()     # store tuples of ((state, path, cost), priority))
    best_costs = {}                     # map states to their best path's cost (lowest cost)

    frontier.push((problem.getStartState(), [], 0), 0)
    best_costs[problem.getStartState()] = 0

    while not frontier.isEmpty():
        curr_state, curr_path, path_cost = frontier.pop()                   # pop the most recent node from the queue (FIFO)
        
        if problem.isGoalState(curr_state): return curr_path                # return path if goal found

        if best_costs.get(curr_state, float('inf')) < path_cost: continue   # if curr_state already has more optimal path, skip
                                                                            # float('inf') used to handle non-existing states
        # process successors
        for state, action, cost in problem.getSuccessors(curr_state):
            new_cost = path_cost + cost
            new_path = curr_path + [action]

            f_score = new_cost + heuristic(state, problem)
    
            if state not in best_costs or new_cost < best_costs[state]:
                best_costs[state] = new_cost  # Update the best known cost
                # Either push new node or update existing one
                frontier.update((state, new_path, new_cost), f_score)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch