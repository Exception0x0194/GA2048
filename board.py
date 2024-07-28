import torch
import random


class Board:
    def __init__(self, size=4):
        self.size = size
        self.board = torch.zeros((size, size), dtype=int)
        self.score = 0
        self.moves_without_operations = 0
        self.add_piece()
        self.add_piece()

    def add_piece(self):
        free_positions = [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if self.board[i][j] == 0
        ]
        if free_positions:
            i, j = random.choice(free_positions)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def slide_and_combine_row(self, row):
        non_zero = row[row != 0]  # remove all zeros
        combined = []
        skip = False
        row_score = 0  # 这一行的得分
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                combined_value = 2 * non_zero[i]
                combined.append(combined_value)
                row_score += combined_value  # 增加得分
                skip = True
            else:
                combined.append(non_zero[i])
        self.score += row_score  # 更新总得分
        return torch.tensor(combined + [0] * (self.size - len(combined)))

    def move(self, direction):
        prev_board = self.board.copy_(self.board)

        if direction == "up":
            self.board = torch.rot90(self.board, 1)
        elif direction == "down":
            self.board = torch.rot90(self.board, -1)
        elif direction == "left":
            self.board = torch.rot90(self.board, 2)

        for i in range(self.size):
            self.board[i] = self.slide_and_combine_row(self.board[i])

        if direction == "up":
            self.board = torch.rot90(self.board, -1)
        elif direction == "down":
            self.board = torch.rot90(self.board, 1)
        elif direction == "left":
            self.board = torch.rot90(self.board, 2)

        self.add_piece()
        if torch.equal(prev_board, self.board):
            self.moves_without_operations += 1
        else:
            self.moves_without_operations = 0

    def game_over(self):
        if torch.any(self.board == 0):
            return False
        if self.moves_without_operations > 3:
            return True
        for i in range(self.size):
            for j in range(self.size):
                if (i < self.size - 1 and self.board[i][j] == self.board[i + 1][j]) or (
                    j < self.size - 1 and self.board[i][j] == self.board[i][j + 1]
                ):
                    return False
        return True

    def get_score(self):
        return self.score


# Test
if __name__ == "__main__":
    board = Board()
    board.board = torch.tensor([[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    print("Initial Board:")
    print(board.board)
    board.move("up")
    print("Board after moving up:")
    print(board.board)
    print("Score:", board.get_score())
    board.board = torch.tensor(
        [[16, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    )
    print(board.board)
    print(f"Game over: {board.game_over()}")
