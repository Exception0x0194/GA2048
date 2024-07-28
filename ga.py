import numpy as np

from net import *
from board import Board


class NeuralNetwork:
    def __init__(self):
        self.model = Net()  # 使用前面定义的神经网络

    def predict(self, input):
        return postprocess(self.model(torch.tensor([preprocess(input)])))

    def mutate(self, mutation_rate=0.01):
        with torch.no_grad():
            for param in self.model.parameters():
                std_dev = param.data.std()  # 计算参数的标准差
                # 基于平均值和标准差调整噪声规模
                noise = torch.randn_like(param) * std_dev * mutation_rate
                param.add_(noise)


def crossover(parent1, parent2):
    child = NeuralNetwork()
    with torch.no_grad():
        for param1, param2, child_param in zip(
            parent1.model.parameters(),
            parent2.model.parameters(),
            child.model.parameters(),
        ):
            child_param.data.copy_((param1.data + param2.data) / 2.0)
    return child


def evaluate(network, games=10):
    total_score = 0
    for _ in range(games):
        total_score += play_game(network)  # 需要实现play_game函数
    return total_score


def evolve(population_size=50, generations=100):
    population = [NeuralNetwork() for _ in range(population_size)]
    for generation in range(generations):
        scores = [evaluate(net) for net in population]
        sorted_tuples = sorted(
            zip(scores, population), key=lambda x: x[0], reverse=True
        )
        sorted_pop = [net for _, net in sorted_tuples]  # 只获取网络实例
        new_population = sorted_pop[: int(0.2 * population_size)]  # 选择表现最好的20%

        # 繁殖和变异
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(new_population, 2, replace=False)
            child = crossover(parent1, parent2)
            child.mutate(mutation_rate=0.1)
            new_population.append(child)

        population = new_population
        print(
            f"Generation {generation} best score: {max(scores)}, avg score {sum(scores)/len(scores)}"
        )

    return population[0]  # 返回表现最好的网络


# 实现一个函数模拟游戏并计算得分
def play_game(network: Net):
    game_board = Board()
    # op_count = 0
    while not game_board.game_over():
        move = postprocess(network.model(preprocess(game_board.board)))
        game_board.move(move)
        # op_count += 1
        # if op_count % 10 == 0:
        #     print(f"{op_count}")
    return game_board.get_score()


# 启动进化过程
if __name__ == "__main__":
    best_network = evolve()
