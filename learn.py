import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import serial
import GetLine

# 하이퍼파라미터 설정
NUM_ACTIONS = 10
STATE_SHAPE = (224, 244, 3)

class StateTransition:
    def __init__(self):
        # 초기 상태 설정
        self.angle = 90
        self.image, self_reward = GetLine.take_picture()

    def get_state(self):
        # 현재 이미지 반환
        return self.image

    def get_reward(self):
        # 기울기 계산 후 반환
        return self.reward

    def move(angle):
        # 모터 회전 후 이동
        self.image, self.reward = GetLine.take_picture()
        return

class Car:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state_transition = StateTransition()
        self.total_reward = 0
        
    def step(self, action):
        self.state_transition(action)
        
        reward = self.get_reward()
        self.total_reward += reward

        return self.state_transition.get_state, reward, (reward == -100)
