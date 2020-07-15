import random
from Grid import Grid
import numpy as np
from BaseAI import BaseAI
import time



WEIGHTS = np.ones((4,4))*4
powers = np.arange(15,-1,-1)
powers[0] = 16
for i in range(4):

	ps = powers[4*i:4*i+4] if i%2 == 0 else powers[4*i:4*i+4][::-1]

	for j in range(4):
		WEIGHTS[i][j] **= ps[j]

MAX_DEPTH = 4


class IntelligentAgent(BaseAI):


	def getMove(self, grid):
		move = self.best_move(grid)
		return move


	def best_move(self, state: Grid):
		'''Returns the best move calculated for Max'''
		start = time.process_time()

		return self.maximize(state, -np.inf, np.inf, start, 0)[0]
		

	def maximize(self, state: Grid, alpha, beta, limit, depth):
		'''the function returns the maximum move and the utility, implementing expectiminimax'''
		#terminate & return utility if its game over, if it exceeds 0.2 seconds, or if it exceeds the maximum depth allowed
		if self.terminal_test(state, time.process_time() - limit, depth):
			return (None, self.utility(state))

		maxUtility = -np.inf

		#order available moves by their utility
		available_moves = state.getAvailableMoves()
		available_moves.sort(key= lambda t: self.utility(t[1]), reverse=True) 

		for (move, grid) in available_moves:

			_, utility = self.chance(grid, alpha, beta, limit, depth)

			if utility > maxUtility:
				maxUtility, maxMove = utility, move
			
			if utility >= beta:
				break

			alpha = max(alpha, utility)

		return maxMove, maxUtility



	def minimize(self, state: Grid, alpha, beta, limit, depth, tile = 2):
		'''
		The function retuns the minimal utility obtain from a state's successors
		Each of the successors is a free spot on the board, in which a new tile can be placed.

		'''
		#terminate & return utility if its game over, if it exceeds 0.2 seconds, or if it exceeds the maximum depth allowed
		if self.terminal_test(state, time.process_time() - limit, depth):
			return (None, self.utility(state))
		
		minMove, minUtility = None, np.inf

		for (x,y) in state.getAvailableCells():
			gridCopy = state.clone()
			gridCopy.insertTile((x,y), tile)
			
			move, utility = self.maximize(gridCopy, alpha, beta, limit, depth + 1)
			
			if utility < minUtility:
				minUtility = utility
				minMove = move
			
			if utility <= alpha:
				break

			beta = min(beta, minUtility)
		
		return minMove, minUtility

	def chance(self, state: Grid, alpha, beta, limit, depth):
		'''A chance node'''
		utility = 0.9*(self.minimize(state, alpha, beta, limit, depth + 1)[1]) + 0.1*(self.minimize(state,alpha,beta,limit, depth + 1, tile = 4)[1])
		return None, utility 

	def terminal_test(self, state:Grid, time, depth):
		return not state.canMove() or time >= 0.199 or depth > MAX_DEPTH


	def weights_heuristic(self, state:Grid):
		
		global WEIGHTS
		
		return sum(WEIGHTS[i][j]*state.map[i][j] for i in range(4) for j in range(4))



	def utility(self, state: Grid):

		if not state.getAvailableCells(): return 0

		h1 = self.weights_heuristic(state)
		h2 = len(state.getAvailableCells())
		
		return h1 + 2**(10-h2)
