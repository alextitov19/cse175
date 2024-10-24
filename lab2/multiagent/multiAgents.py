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

import random
from util import manhattanDistance
from game import Directions

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        Returns the best action according to the evaluation function.
        """
        # Collect legal moves
        legalMoves = gameState.getLegalActions()

        # Evaluate each action
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        This evaluation function looks at the game state after Pacman takes an action
        and returns a score that encourages Pacman to eat food, avoid ghosts, and chase
        scared ghosts.
        """
        # Generate the successor state
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Start with the game state's base score
        score = successorGameState.getScore()

        # Evaluate distance to the closest food
        foodList = newFood.asList()
        if foodList:
            closestFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            score += 10 / (closestFoodDistance + 1)  # Use reciprocal for proximity bonus

        # Evaluate ghost distances and scared times
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)

            if scaredTime > 0:
                # If ghost is scared, chase it
                score += 10 / (distanceToGhost + 1)
            else:
                # If ghost is not scared, avoid it
                if distanceToGhost < 2:
                    score -= 100  # Heavily penalize for being too close to a ghost
                else:
                    score -= 5 / (distanceToGhost + 1)  # Penalize being close to a ghost

        # Penalize stopping (Pacman should avoid staying still)
        if action == Directions.STOP:
            score -= 10

        return score

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        # Call the recursive minimax function starting from Pacman's turn (agentIndex=0) and depth=0
        best_action = self.minimax(gameState, 0, 0)
        return best_action[1]  # Return the action with the best score

    def minimax(self, gameState, depth, agentIndex):
        """
        Recursive minimax helper function.
        Returns a tuple (score, action), where score is the minimax value of the state and action
        is the best action to take from that state.
        """
        # If we reach maximum depth or a terminal state (win or lose), evaluate the state
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # Determine whether the agent is Pacman (maximizing) or a ghost (minimizing)
        if agentIndex == 0:  # Pacman (Max)
            return self.maxValue(gameState, depth)
        else:  # Ghost (Min)
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth):
        """
        Maximizing function for Pacman (agentIndex=0).
        """
        legalActions = gameState.getLegalActions(0)  # Pacman is agentIndex 0
        if not legalActions:
            return self.evaluationFunction(gameState), None

        best_score = float('-inf')
        best_action = None

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score, _ = self.minimax(successor, depth, 1)  # Call minimax for the first ghost
            if score > best_score:
                best_score, best_action = score, action

        return best_score, best_action

    def minValue(self, gameState, depth, agentIndex):
        """
        Minimizing function for ghosts (agentIndex >= 1).
        """
        legalActions = gameState.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(gameState), None

        best_score = float('inf')
        best_action = None

        nextAgent = agentIndex + 1  # Move to the next agent
        if nextAgent == gameState.getNumAgents():  # If all agents moved, go back to Pacman and increase depth
            nextAgent = 0

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if nextAgent == 0:  # Pacman's turn next
                score, _ = self.minimax(successor, depth + 1, nextAgent)
            else:  # Another ghost's turn next
                score, _ = self.minimax(successor, depth, nextAgent)

            if score < best_score:
                best_score, best_action = score, action

        return best_score, best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        #Used only for pacman agent hence agentindex is always 0.
        def maxLevel(gameState,depth,alpha, beta):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   #Terminal Test 
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            alpha1 = alpha
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,minLevel(successor,currDepth,1,alpha1,beta))
                if maxvalue > beta:
                    return maxvalue
                alpha1 = max(alpha1,maxvalue)
            return maxvalue
        
        #For all ghosts.
        def minLevel(gameState,depth,agentIndex,alpha,beta):
            minvalue = 999999
            if gameState.isWin() or gameState.isLose():   #Terminal Test 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            beta1 = beta
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents()-1):
                    minvalue = min (minvalue,maxLevel(successor,depth,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
                else:
                    minvalue = min(minvalue,minLevel(successor,depth,agentIndex+1,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
            return minvalue

        # Alpha-Beta Pruning
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        alpha = -999999
        beta = 999999
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a min level. Hence calling min for successors of the root.
            score = minLevel(nextState,0,1,alpha,beta)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
            # Updating alpha value at root.    
            if score > beta:
                return returnAction
            alpha = max(alpha,score)
        return returnAction

        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

