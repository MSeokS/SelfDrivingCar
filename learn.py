import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

# 하이퍼파라미터 설정
num_actions = 2
state_size = 4
gamma = 0.99  # 할인 계수
epsilon = 1.0  # 초기 탐험 확률
epsilon_min = 0.01  # 최소 탐험 확률
epsilon_decay = 0.995  # 탐험 확률 감소율
learning_rate = 0.001
batch_size = 64
memory_size = 2000

# Q-네트워크 모델 정의
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=state_size, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

# 리플레이 메모리 클래스 정의
class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

# DQN 에이전트 클래스 정의
class DQNAgent:
    def __init__(self):
        self.model = build_model()
        self.target_model = build_model()
        self.update_target_model()
        self.memory = ReplayMemory(memory_size)
        self.epsilon = epsilon

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(num_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + gamma * np.amax(t[0])
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

# 환경 설정 및 학습
env = gym.make('CartPole-v1')
agent = DQNAgent()

episodes = 1000
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.memory.add((state, action, reward, next_state, done))
        state = next_state
        if done:
            agent.update_target_model()
            print(f"에피소드: {e+1}/{episodes}, 점수: {time}, 탐험 확률: {agent.epsilon:.2}")
            break
        agent.train(batch_size)

env.close()
