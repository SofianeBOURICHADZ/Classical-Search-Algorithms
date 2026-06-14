import random
import copy

# ------------------------------------------------------------------------------
# TicTacToe Game Class
# ------------------------------------------------------------------------------

class TicTacToe:
    def __init__(self):
        self.size = 9
        self.board = [[' ' for i in range(0,3)] for j in range(0,3)]
        self.human = -1        # Human player identifier.
        self.comp = 1          # Computer player identifier.
        # Symbols to represent players on the board.
        self.h_symbol = 'X'
        self.c_symbol = 'O'
        # Initial depth is the total number of moves possible.
        self.depth = self.size

    def goal_test(self, state, player):
        symbol = self.c_symbol if player == self.comp else self.h_symbol

        # Check rows and columns.
        for i in range(3):
            if all(state[i][j] == symbol for j in range(3)):
                return True
            if all(state[j][i] == symbol for j in range(3)):
                return True

        # Check diagonals.
        if state[0][0] == symbol and state[1][1] == symbol and state[2][2] == symbol:
            return True
        if state[0][2] == symbol and state[1][1] == symbol and state[2][0] == symbol:
            return True

        return False

    def game_over(self, state):
        return self.goal_test(state, self.human) or self.goal_test(state, self.comp) or len(self.empty_cells(state)) == 0

    def empty_cells(self, state):
        cells = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == ' ':
                    cells.append((i, j))
        return cells

    def valid_move(self, x, y):
        if 0 <= x and x <= 2 and 0 <= y and y <= 2:
            return True if self.board[x][y] == ' ' else False
        return False

    def set_move(self, x, y, player):
        if self.valid_move(x,y):
            self.board[x][y] = self.h_symbol if player == self.human else self.c_symbol
            self.depth =- 1
            return True
        return False

    def render(self, state):
        print("\n-------")
        for row in state:
            print("|" + "|".join(row) + "|")
        print("-------\n")

    def human_turn(self):
        while(True):
            try:
              x = int(input('Enter X coordinate (number only)'))
              y = int(input('Enter Y coordinate (number only)'))
              if self.valid_move(x,y):
                self.set_move(x,y,self.human)
                return (x,y)
              print('Invalid Move pls retry coords out of bounds')
            except ValueError:
                print('ENter integer')

    def evaluate(self, state):
        h = self.goal_test(self.board,self.human)
        c = self.goal_test(self.board,self.comp)
        if h == True:
            return -1
        if c ==True:
            return 1
        return 0

    def run(self):
        agent = MiniMaxAgent(self)
        starter = self.human
        is_human = True
        while(self.game_over(self.board) == False):
            self.render(self.board)
            if is_human:
                is_human = False
                self.human_turn()
            else:
                is_human = True
                move = agent.play()
                print(move)
                self.set_move(move[0],move[1],self.comp)
            self.depth -= 1
        self.render(self.board)
        print(self.evaluate(self.board))

        

# ------------------------------------------------------------------------------
# MiniMaxAgent Class using the MiniMax Algorithm
# ------------------------------------------------------------------------------

class MiniMaxAgent:
    def __init__(self, game):
        self.game = game
    def minimax(self, state, depth, player):
        if depth == 0 or self.game.game_over(state) == True:
            return [-1, -1, self.game.evaluate(state)]
        best_move = [-1,-1,float('inf')] if player == self.game.human else [-1,-1,-float('inf')]
        empty_cells = self.game.empty_cells(state)
        st = copy.deepcopy(state)
        for cell in empty_cells:
            if self.game.set_move(cell[0],cell[1],player):
              move = self.minimax(self.game.board,depth - 1, -player)
              if (self.game.human == player and move[2] <=best_move[2]) or (self.game.comp == player and move[2] >= best_move[2]):
                  best_move[0]=move[0]
                  best_move[1]=move[1]
                  best_move[2]=move[2]
        self.game.board = st
        return best_move
    def play(self):
        if len(self.game.empty_cells(self.game.board)) == 9:            
            x, y = random.choice([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)])
        else:
            depth = self.game.depth
            move = self.minimax(self.game.board, depth, self.game.comp)
            x, y = move[0], move[1]
        return (x, y)

# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    game = TicTacToe()
    game.run()