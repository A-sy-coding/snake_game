import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 80

class AISnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # play_step단계가 가장 중요하다.

    def play_step(self, action): # action 인자를 추가한다.

        self.frame_iteration += 1  # frame_iteration이 계속 커질수 있게 설정한다.

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0 # 보상 값 초기화
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake): # 부딪히거나 오랜시간동안 향상되지 않으면 멈춘다.
            game_over = True                                                   # 뱀이 길어질수록 더 긴 시간을 갖는다.
            reward = -10  # game over 하면 보상을 -10을 준다.
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10 # food를 먹으면 10을 준다.
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt = None): # point(pt)를 none으로 둔다.
                                       # is_collision은 public이여서 앞에 _를 제거한다.

        if pt is None:   # pt에 값이 없으면 위에서 저장한 head point를 pt에 저장한다.
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):

        # action -> [straight, right, left]
        # Direction -> RIGHT = 1 , LEFT = 2, UP = 3,  DOWN = 4

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction) # 현재 direction은 clock_wise안에 존재한다.

        if np.array_equal(action, [1,0,0]):  # action값과 [1,0,0]이 같으면 true를 반환한다.
            new_dir = clock_wise[idx]  # 변하지 않는다. -> 계속 직진이면 direction도 계속 유지된다.
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right => right -> down -> left -> up
                                           # 오른쪽으로 이동하면 direction도 계속 바뀐다.
        else: # action = [0,0,1]
            next_idx = (idx - 1) % 4 # 위와 다르게 반대편으로 이동한다.
            new_dir = clock_wise[next_idx]  # left => right -> up -> left -> down
                                            # 왼쪽으로 이동하면 direction도 반대로 계속 바뀐다.
        
        self.direction = new_dir  # 이동한 방향으로 저장

        x = self.head.x   # head의 x point 값
        y = self.head.y   # head의 y point 값
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE  # 방향이 오른쪽이면 x값이 block_size(20)만큼 증가
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE  # 방향이 왼쪽이면 x값이 block_size(20)만큼 감소
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE  # 방향이 밑이면 y값이 block_size(20)만큼 증가
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE  # 방향이 위이면 y값이 block_size(20)만큼 감소

        self.head = Point(x, y)
            

