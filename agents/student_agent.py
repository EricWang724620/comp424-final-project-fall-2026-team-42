"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, LINES, state_key)
from constants import *
import numpy as np
import time

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self.time_margin = 0.15          # leave safety margin, could be change later
        self.max_depth = 3  # maximum search depth, could be change later
        # self.cache = {}  # cache for memoization, may use in the future for speed imporvement

    def evaluate(self, board, reserves, player):
        """Simple evaluation: count own almost-wins and block opponent's."""
        # this version does not include factor of useable pieces, heuristic function etc.. could be furthur improved
        owners, sizes = top_view(board) # who owns the top piece at (r,c): 0=empty, 1 or 2=player, 3=obstacle
        score = 0
        opp = 3 - player # player number is 1 or 2, 3 - our number is the opponent number
        for line in LINES[WIN_LENGTH]: # Loop over every possible 4 in a row line on the board
            own = opp_count = empty = 0
            for r, c in line:       # r,c refers to row, column respectively
                cell = owners[r, c]
                if cell == player:
                    own += 1
                elif cell == opp:
                    opp_count += 1
                elif cell == 0:
                    empty += 1
                # cell == 3 is obstacle this line is dead, ignore
            if own == WIN_LENGTH: # we win
                score += 10000
            elif opp_count == WIN_LENGTH: # opponent win
                score -= 10000
            elif own == WIN_LENGTH - 1 and empty == 1: # we almost win
                score += 800
            elif opp_count == WIN_LENGTH - 1 and empty == 1: # opponent almost win
                score -= 700
            elif own == WIN_LENGTH - 2 and empty == 2: # we have a 2 in a row
                score += 60
            elif opp_count == WIN_LENGTH - 2 and empty == 2: score -= 50 # opponent have a 2 in a row

        # 2. Large piece control
        # the larger the piece, more hard to cover it, award pieces controlling by their size on the board
        for r in range(4):
            for c in range(4):
                if owners[r, c] == player and sizes[r, c] == LARGE:
                    score += 40
                elif owners[r, c] == opp and sizes[r, c] == LARGE:
                    score -= 40
        # 3. Reserves
        # the pieces still in hands, also award by their size, but this time is smaller than the on board
        # since it is not controlling anything yet, and desired block may already controlled by opponent
        score += 30 * reserves[player][LARGE] + 15 * reserves[player][MEDIUM] + 5 * reserves[player][SMALL]
        score -= 30 * reserves[opp][LARGE] + 15 * reserves[opp][MEDIUM] + 5 * reserves[opp][SMALL]
        return score
    def _minimax(self, board, reserves, player, opponent, depth, deadline,
                 alpha, beta, maximizing):
        if time.perf_counter() >= deadline: # stop when close to time limit 2s, our design it to stop at 1.85, leave some space for rest of the program
            return None, 0

        if depth == 0: # we reached the depth we desired, iterative deepening
            return None, self.evaluate(board, reserves, player)

        moves = get_valid_moves(board, reserves, player if maximizing else opponent) # Generate legal moves for the side that is about to play, support by the helper
        if not moves: # no moves left, so a very bad state, award if we can force opponent get there, penalty if we move there
            return None, -1000 if maximizing else 1000

        best_move = None
        if maximizing:
            best_val = -np.inf
            for mv in moves:
                b, r = copy_state(board, reserves) # make a copy and play/try on the copy
                execute_move(b, r, player, mv)      # b-> board, r->reserves, mv->move
                # check if there is a move that make us immediate win
                if check_winner(b, player) == player:
                    return mv, 10000
                _, val = self._minimax(b, r, player, opponent, depth-1, deadline,
                                       alpha, beta, False) # recursing, its opponent's turn 
                if val > best_val: # keep the move that gives the highest score
                    best_val = val
                    best_move = mv
                alpha = max(alpha, best_val)
                if alpha >= beta: # alpha beta pruning
                    break
            return best_move, best_val
        else:
            best_val = np.inf # similar idea for opponent turn
            for mv in moves:
                b, r = copy_state(board, reserves)
                execute_move(b, r, opponent, mv)
                if check_winner(b, opponent) == opponent:
                    return mv, -10000
                _, val = self._minimax(b, r, player, opponent, depth-1, deadline,
                                       alpha, beta, True)
                if val < best_val:
                    best_val = val
                    best_move = mv
                beta = min(beta, best_val)
                if alpha >= beta:
                    break
            return best_move, best_val
    
    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        # start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.
        # return random_move(board, reserves, player)
        start = time.perf_counter()
        deadline = start + STEP_TIME_LIMIT - self.time_margin # when to stop search

        moves = get_valid_moves(board, reserves, player) # if no legal move at all, use default, but could just replace by None
        if not moves:
            return random_move(board, reserves, player)

        opponent_pieices = np.count_nonzero(board == opponent)
        # Quick win / avoid immediate loss (like greedy_agent)
        safe_moves = []
        for mv in moves:
            b, r = copy_state(board, reserves)
            execute_move(b, r, player, mv)
            w = check_winner(b, player)
            if w == player:
                return mv
            if w == opponent:
                continue  # this move uncovers opponent's win

            #Check if the opponent do one more move can the oponent win 
            #if no we don't even need to simulate the possible outcomes because either way its number of pieces falls short

            if opponent_pieices + 1 < WIN_LENGTH:
                safe_moves.append(mv)
                continue
            # check if opponent can win immediately after
            opp_can_win = False
            for om in get_valid_moves(b, r, opponent): # om as opponent move
                b2, r2 = copy_state(b, r)
                execute_move(b2, r2, opponent, om)
                if check_winner(b2, opponent) == opponent:
                    opp_can_win = True
                    break
            if not opp_can_win: # Only keep moves that do not let the opponent win immediately
                safe_moves.append(mv)

        candidates = safe_moves or moves # the or moves is for situation: if every move will let opponent win, even we don't want, we have to play something
        
        # make the agent avoid running uncessary moves when you have one safest move or only on move
        if(len(candidates) == 1):
            return candidates[0]
        
        # iterative deepening with time control
        best_move = candidates[0] # initialize so even we run out of time, we still play something
        for depth in range(1, self.max_depth + 1):
            if time.perf_counter() >= deadline: # if times almost up, stop search
                break
            move, val = self._minimax(  # run our minimax then return best move found within time limit
                board, reserves, player, opponent, depth, deadline,
                alpha=-np.inf, beta=np.inf, maximizing=True
            )
            if move is not None:
                best_move = move

        return best_move
