import numpy as np

class Board():
    def __init__(self):
        self.board = [0 for i in range(100)]
        self.board[44] = 1
        self.board[45] = -1
        self.board[54] = -1
        self.board[55] = 1
        self.dir = [-11,-10,-9,-1,1,9,10,11]

    def step(self, pos, player):
        self.board[pos] = player
        for d in self.dir:
            self.flip(pos, d, player)

    def flip(self, pos, d, player):
        bracket = self.causeFlip(pos, d, player)
        if bracket >= 0:
            p = pos + d
            while p != bracket:
                self.board[p] = player
                p += d

    def ableFlip(self, pos, player):
        if self.board[pos] != 0:
            return False
        for d in self.dir:
            if self.causeFlip(pos, d, player) >= 0:
                return True
        return False

    def causeFlip(self, pos, d, player):
        c = pos + d
        if self.board[c] != player * -1:
            return -1
        return self.findBracket(c + d, d, player)

    def findBracket(self, pos, d, player):
        if self.board[pos] == player:
            return pos
        elif self.board[pos] == player * -1:
            return self.findBracket(pos + d, d, player)
        else:
            return -1

    def render(self):
        for i in range(1,9):
            c = ""
            for j in range(1,9):
                player = self.board[i*10+j]
                if player == 1:
                    c += "● "
                elif player == -1:
                    c += "○ "
                else:
                    c += "- "
            print(c)
        print()

    def validPos(self, pos):
        k = pos % 10
        if 11 <= pos and pos <= 88 and k > 0 and k < 9:
            return True
        else:
            return False

    def candidates(self, player):
        candidates = []
        for p in range(11,89):
            if self.validPos(p) and self.ableFlip(p,player):
                candidates.append(p)
        return candidates

    def finished(self):
        if candidates(1) == 0 and candidates(-1) == 0:
            return True
        else:
            return False

    def convert(self, player):
        board = np.zeros(shape=(2,8,8), dtype=np.float32)
        for y in range(1,9):
            for x in range(1,9):
                p = self.board[y*10+x]
                if p == player:
                    board[0][y-1][x-1] = 1.0
                else:
                    board[1][y-1][x-1] = 1.0
        return board

    def winner(self):
        s = sum(self.board)
        if s > 0:
            return 1
        elif s < 0:
            return -1
        else:
            return 0
