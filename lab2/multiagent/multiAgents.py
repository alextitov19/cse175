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
          Returns the minimax action using self.depth and self.evaluationFunction.
        """
        # Start the minimax algorithm with alpha-beta pruning
        alpha = float('-inf')
        beta = float('inf')
        best_action = self.alphaBeta(gameState, 0, 0, alpha, beta)
        return best_action[1]

    def alphaBeta(self, gameState, depth, agentIndex, alpha, beta):
        """
        Recursive helper function for Alpha-Beta pruning.
        Returns a tuple (score, action), where score is the minimax value of the state
        and action is the best action to take from that state.
        """
        # If the depth is reached or the game has ended, evaluate the state
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # Determine whether the agent is Pacman (maximizing) or a ghost (minimizing)
        if agentIndex == 0:  # Pacman (Max)
            return self.maxValue(gameState, depth, alpha, beta)
        else:  # Ghost (Min)
            return self.minValue(gameState, depth, agentIndex, alpha, beta)

    def maxValue(self, gameState, depth, alpha, beta):
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
            score, _ = self.alphaBeta(successor, depth, 1, alpha, beta)  # Call alpha-beta for the first ghost

            if score > best_score:
                best_score = score
                best_action = action

            # Update alpha and check for pruning
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta cutoff

        return best_score, best_action

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
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
            if nextAgent == 0:  # Pacman's turn next, so increase depth
                score, _ = self.alphaBeta(successor, depth + 1, nextAgent, alpha, beta)
            else:  # Another ghost's turn next
                score, _ = self.alphaBeta(successor, depth, nextAgent, alpha, beta)

            if score < best_score:
                best_score = score
                best_action = action

            # Update beta and check for pruning
            beta = min(beta, score)
            if alpha >= beta:
                break  # Alpha cutoff

        return best_score, best_action
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning (question 3).
    """

    def getAction(self, gameState):
        """
        Returns the optimal action for Pacman using alpha-beta pruning and the minimax algorithm.
        """

        # Function to handle Pacman's move (maximizing player)
        def maximizeValue(state, depth, alpha, beta):
            nextDepth = depth + 1
            if state.isWin() or state.isLose() or nextDepth == self.depth:
                return self.evaluationFunction(state)

            best_value = float('-inf')
            available_moves = state.getLegalActions(0)
            local_alpha = alpha

            # Loop through each of Pacman's legal actions
            for move in available_moves:
                newState = state.generateSuccessor(0, move)
                best_value = max(best_value, minimizeValue(newState, nextDepth, 1, local_alpha, beta))
                if best_value > beta:  # Beta pruning
                    return best_value
                local_alpha = max(local_alpha, best_value)
            return best_value

        # Function to handle ghost's move (minimizing players)
        def minimizeValue(state, depth, agentIndex, alpha, beta):
            min_value = float('inf')
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            available_moves = state.getLegalActions(agentIndex)
            local_beta = beta

            # Loop through each of the ghost's legal actions
            for move in available_moves:
                newState = state.generateSuccessor(agentIndex, move)
                if agentIndex == state.getNumAgents() - 1:  # Last ghost; Pacman moves next
                    min_value = min(min_value, maximizeValue(newState, depth, alpha, local_beta))
                else:  # Another ghost moves next
                    min_value = min(min_value, minimizeValue(newState, depth, agentIndex + 1, alpha, local_beta))

                if min_value < alpha:  # Alpha pruning
                    return min_value
                local_beta = min(local_beta, min_value)

            return min_value

        # Alpha-Beta Pruning logic starts here
        chosen_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # Evaluate all legal actions for Pacman
        for move in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, move)
            score = minimizeValue(next_state, 0, 1, alpha, beta)

            # Track the best action and score
            if score > best_score:
                chosen_action = move
                best_score = score

            # Update alpha at the root
            if score > beta:
                return chosen_action
            alpha = max(alpha, score)

        return chosen_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent that calculates expected value for ghosts and maximizes for Pacman.
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        # Start the expectimax search with Pacman (agentIndex = 0)
        best_score, best_action = self.expectimax(gameState, 0, 0)
        return best_action

    def expectimax(self, gameState, depth, agentIndex):
        """
        Expectimax helper function that returns the best score and corresponding action.
        """
        # If it's a terminal state or max depth, return the evaluation function
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        # Pacman's turn (maximizing)
        if agentIndex == 0:
            return self.maxValue(gameState, depth)

        # Ghosts' turn (chance node)
        else:
            return self.expectValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth):
        """
        Maximizing function for Pacman (agentIndex = 0).
        """
        best_score = float('-inf')
        best_action = None

        # Evaluate all possible actions for Pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score, _ = self.expectimax(successor, depth, 1)  # Call expectimax for ghost's turn
            if score > best_score:
                best_score = score
                best_action = action

        return best_score, best_action

    def expectValue(self, gameState, depth, agentIndex):
        """
        Expected value function for ghosts (chance node).
        """
        # Get the legal actions for the ghost
        actions = gameState.getLegalActions(agentIndex)
        num_actions = len(actions)

        # If no legal actions, return terminal evaluation
        if num_actions == 0:
            return self.evaluationFunction(gameState), None

        expected_score = 0

        # Iterate through all actions for the ghost
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)

            # If it's the last ghost, go back to Pacman and increase the depth
            if agentIndex == gameState.getNumAgents() - 1:
                score, _ = self.expectimax(successor, depth + 1, 0)
            else:
                # Otherwise, continue with the next ghost
                score, _ = self.expectimax(successor, depth, agentIndex + 1)

            # Calculate the expected value (average over all possible actions)
            expected_score += score / num_actions

        return expected_score, None


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

