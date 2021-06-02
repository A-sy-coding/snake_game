import torch
import random
import numpy as np
from collections import deque # 찾아보기
from snake_game import AISnakeGame, Direction, Point # snake_game에서의 class 불러오기
from model import Linear_QNet, QTrainer  # model.py에서 불러오기
from helper import plot

MAX_MEMORY = 100_000  # 저장할 수 있는 메모리 지정
BATCH_SIZE = 1000
LR = 0.001  # learning rate

class Agent:

    def __init__(self):
        self.n_games = 0 # 게임을 한 횟수
        self.epsilon = 0 # randomness를 조절한다.
        self.gamma = 0.9  # discount rate  # 1보다 작아야 한다.
        self.memory = deque(maxlen = MAX_MEMORY) # 이 memory를 초과하면 삭제한다.-> popleft()
        self.model = Linear_QNet(11, 256, 3)  # 입력층은 11개이고 출력층은 3개이다.
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)


    def get_state(self, game):
        head = game.snake[0] # head 값 저장
        point_l = Point(head.x - 20, head.y) # 왼쪽으로 이동하면 x값이 block_size(20)만큼 이동
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT  # 현재의 방향이 LEFT와 같으면 TRUE(1)
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or  # 우측 방향이고 우측으로 이동했을 때 부딪히면 1 아니면 0을 출력
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or  # 위쪽 방향이고 우측으로 이동했을 때 부딪히면 1 아니면 0을 출력
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or  # 아래쪽 방향이고 우측으로 이동했을 때 부딪히면 1 아니면 0을 출력
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) ,

            # move direction
            dir_l, #   True, False 형태로 나온다.
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x,  # food가 왼쪽에 있다.
            game.food.x > game.head.x,  # food가 오른쪽에 있다.
            game.food.y < game.head.y,  # food가 위쪽에 있다.
            game.food.y > game.head.y  # food가 아래쪽에 있다.
        ]

        return np.array(state, dtype=int)  # state값들이 0 또는 1 값으로 저장된다.



    def remember(self, state, action, reward, next_state, done):
        self.memory.append( (state, action, reward, next_state, done) ) # 메모리를 초과하면 popleft된다.


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuple

        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # 위의 2줄 코드와 의미가 동일하다.
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)



    def train_short_memory(self, state, action, reward, next_state, done):  # 한 게임만을 저장한다.
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random move
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]

        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # pytorch에는 predict API가 없다.
            move = torch.argmax(prediction).item() # 예측값 중 가장 큰 값의 인덱스를 저장
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    agent = Agent()  # Agent 클래스 객체 생성
    game = AISnakeGame()  # AISnakeGame 클래스 객체 생성
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perfom move and get new state
        reward, done, score = game.play_step(final_move) # 이동을 게임에 적용한다.
        state_new = agent.get_state(game) # 바뀐 새로운 상태를 저장한다.

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:  # 게임이 끝나면

            # train long memory, plot result
            game.reset()
            agent.n_games += 1  # 게임 반복수를 늘려나간다.
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game',agent.n_games, 'Score',score, 'Record',record)

            # plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)



if __name__ == '__main__':
    train()