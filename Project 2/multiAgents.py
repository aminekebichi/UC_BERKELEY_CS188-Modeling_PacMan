# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        # Get closest food distance
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            score += 10.0 / minFoodDistance
        
        # Avoid ghosts or chase them if scared
        for i, ghostState in enumerate(newGhostStates):
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            if newScaredTimes[i] > 0:
                score += 100.0 / (ghostDistance + 1)
            else:
                if ghostDistance < 2:
                    score -= 500
        
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # minimax function
        def minimax(state, depth, agent):

            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            # Pacmans turn (maximize)
            if agent == 0:
                best = float('-inf')
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    score = minimax(next_state, depth, 1)
                    best = max(best, score)
                return best
            
            # Ghosts turn (minimize)
            else:
                best = float('inf')
                next_agent = agent + 1
                next_depth = depth
                
                if next_agent == state.getNumAgents():
                    next_agent = 0
                    next_depth = depth - 1
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    score = minimax(next_state, next_depth, next_agent)
                    best = min(best, score)
                return best

        best_action = None
        best_score = float('-inf')
        
        for action in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, action)
            score = minimax(next_state, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        
        # alphabeta function
        def alphabeta(state, depth, agent, alpha, beta):

            # base case
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            # pacmans turn (maximize)
            if agent == 0:
                best = float('-inf')
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    score = alphabeta(next_state, depth, 1, alpha, beta)
                    best = max(best, score)
                    alpha = max(alpha, best)
                    if best > beta:
                        return best
                return best
            
            # ghosts turn (minimize)
            else:
                best = float('inf')
                next_agent = agent + 1
                next_depth = depth
                
                if next_agent == state.getNumAgents():
                    next_agent = 0
                    next_depth = depth - 1
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    score = alphabeta(next_state, next_depth, next_agent, alpha, beta)
                    best = min(best, score)
                    beta = min(beta, best)
                    if best < alpha:
                        return best
                return best
        
        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, action)
            score = alphabeta(next_state, self.depth, 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)
        
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        
        # expectimax function
        def expectimax(state, depth, agent):

            # base case
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            # pacmans turn (maximize)
            if agent == 0:
                best = float('-inf')
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    score = expectimax(next_state, depth, 1)
                    best = max(best, score)
                return best
            
            # ghosts turn (expect average)
            else:
                total = 0
                actions = state.getLegalActions(agent)
                next_agent = agent + 1
                next_depth = depth
                
                if next_agent == state.getNumAgents():
                    next_agent = 0
                    next_depth = depth - 1
                for action in actions:
                    next_state = state.generateSuccessor(agent, action)
                    score = expectimax(next_state, next_depth, next_agent)
                    total += score
                return total / len(actions)
        
        best_action = None
        best_score = float('-inf')
        
        for action in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, action)
            score = expectimax(next_state, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    Focuses on eating food while avoiding ghosts. Rewards being close to food
    and penalizes being close to active ghosts. Chases scared ghosts for bonus points.
    """
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    
    # get closest food
    if food:
        closest_food = min([manhattanDistance(pos, f) for f in food])
        score += 10.0 / closest_food
    
    # ghosts
    for ghost in ghosts:
        dist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            if dist > 0:
                score += 200.0 / dist
        else:
            if dist < 2:
                score -= 500

    score -= 4 * len(food)
    
    return score

# Abbreviation
better = betterEvaluationFunction