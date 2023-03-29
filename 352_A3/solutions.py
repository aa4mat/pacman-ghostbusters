# solutions.py
# ------------
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

'''Implement the methods from the classes in inference.py here'''

import util
from util import raiseNotDefined
import random
import busters


def normalize(self):
    """
    Normalize the distribution such that the total value of all keys sums
    to 1. The ratio of values for all keys will remain the same. In the case
    where the total value of the distribution is 0, do nothing.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> dist.normalize()
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
    >>> dist['e'] = 4
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
    >>> empty = DiscreteDistribution()
    >>> empty.normalize()
    >>> empty
    {}
    """
    # *** YOUR CODE HERE ***"
    # if the distribution is empty then do nothing
    if len(self.keys()) == 0:
        return None
    # If the total value of the distribution is 0, do nothing
    if self.total() == 0:
        return None
    # Normalize the distribution total value of all keys sums to 1
    factor = 1.0 / self.total()
    for k in self:
        self[k] = self[k] * factor


def sample(self):
    """
    Draw a random sample from the distribution and return the key, weighted
    by the values associated with each key.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> N = 100000.0
    >>> samples = [dist.sample() for _ in range(int(N))]
    >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
    0.2
    >>> round(samples.count('b') * 1.0/N, 1)
    0.4
    >>> round(samples.count('c') * 1.0/N, 1)
    0.4
    >>> round(samples.count('d') * 1.0/N, 1)
    0.0
    """
    # *** YOUR CODE HERE ***
    # if the distribution is empty then do nothing
    if len(self.keys()) == 0:
        return None
    # If the total value of the distribution is 0, do nothing
    if self.total() == 0:
        return None

    # If the distribution is not normalized.
    if self.total() != 1:
        normalize(self)
    # Set a random float value
    rand_val = random.random()
    # Draw a random sample from the distribution
    total = 0
    for k, v in self.items():
        total += v
        if rand_val <= total:
            return k


def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition,
                       jailPosition):
    """
    Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
    """
    "*** YOUR CODE HERE ***"
    if ghostPosition == jailPosition and noisyDistance is None:
        return 1
    elif ghostPosition == jailPosition or noisyDistance is None:
        return 0

    return busters.getObservationProbability(noisyDistance,
                                             util.manhattanDistance(
                                                 pacmanPosition, ghostPosition))


def observeUpdate(self, observation, gameState):
    """
    Update beliefs based on the distance observation and Pacman's position.

    The observation is the noisy Manhattan distance to the ghost you are
    tracking.

    self.allPositions is a list of the possible ghost positions, including
    the jail position. You should only consider positions that are in
    self.allPositions.

    The update model is not entirely stationary: it may depend on Pacman's
    current position. However, this is not a problem, as Pacman's current
    position is known.
    """
    "*** YOUR CODE HERE ***"

    for position in self.allPositions:
        belief = self.beliefs[position] * self.getObservationProb(
            observation, gameState.getPacmanPosition(),
            position, self.getJailPosition())
        # P(ghost) * P(observation | ghost, pacman)

        # new belief = P(ghost | observation)
        self.beliefs[position] = belief
        # update belief for each position

    self.beliefs.normalize()


def elapseTime(self, gameState):
    """
    Predict beliefs in response to a time step passing from the current
    state.
    The transition model is not entirely stationary: it may depend on
    Pacman's current position. However, this is not a problem, as Pacman's
    current position is known.
    """
    "*** YOUR CODE HERE ***"
    numAgents = gameState.getNumAgents()  # number of agents
    # print("number of agents (2?) = {}".format(numAgents))
    for ghostAgentIndex in range(1, numAgents):
        # Get the previous (current) ghost position
        oldPos = gameState.getGhostPosition(ghostAgentIndex)
        # pacMan is agentIndex 0, the first ghost is agentIndex 1 and so on.
        # for multiple ghosts, loop over number of ghosts.

        # error for ghost - ghost is None?
        # line 279, in getGhostPosition
        # ***     return self.data.agentStates[agentIndex].getPosition()
        # *** AttributeError: 'NoneType' object has no attribute 'getPosition'

        # Get the distribution over new positions for the ghost,
        # given its previous position
        # P(newPos | oldPos)
        newPosDist = self.getPositionDistribution(gameState, oldPos)
        # Update the belief at every position on the map
        for p in newPosDist:
            observeUpdate(self, newPosDist[p], gameState)
