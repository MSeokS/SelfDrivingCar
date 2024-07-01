import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import serial
import GetLine
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import sys
import time

# 하이퍼파라미터 설정
WIDTH = 640
HEIGHT = 360
MIN_ANGLE = 10
MAX_ANGLE = 30

NUM_ACTIONS = 10
STATE_SHAPE = (HEIGHT, WIDTH, 3)
GAMMA = 0.5
BATCH_SIZE = 32
MIN_REPLAY_SIZE = 100
REPLAY_SIZE = 2000
TARGET_UPDATE_FREG = 1

MAX_EPISODE = 100

EPSILON = 0.99
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

class StateTransition:
    def __init__(self, cap, arduino):
        # 초기 상태 설정
        self.cap = cap
        self.arduino = arduino
        self.image, self.reward = self.take_picture()

    def get_state(self):
        # 현재 이미지 반환
        return self.image

    def get_reward(self):
        # 리워드 반환
        return self.reward

    def move(self, action):
        # 모터 회전 후 이동
        angle = ((MAX_ANGLE - MIN_ANGLE) / (NUM_ACTIONS - 1)) * action + MIN_ANGLE
        self.arduino.write(bytes(f"{angle}\n", 'utf-8'))
        self.image, self.reward = self.take_picture()
        return
    
    def stop(self):
        self.arduino.write(bytes("0", 'utf-8'))

    def take_picture(self):
        image, reward = GetLine.take_picture(self.cap)
        if image is None:
            print("Camera disconnect.")
            sys.exit()
        else:
            return image, reward

class Car:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)
        self.reset()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'arduino'):
            self.arduino.close()

    def reset(self):
        self.state_transition = StateTransition(self.cap, self.arduino)
        self.total_reward = 0
        return self.state_transition.get_state()
        
    def step(self, action):
        self.state_transition.move(action)
        
        reward = self.state_transition.get_reward()
        self.total_reward += reward

        return self.state_transition.get_state(), reward, (reward == -100)

class DQNAgent:
    def __init__(self):
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.min_replay_memory_size = MIN_REPLAY_SIZE
        self.target_update_freq = 1

        self.model = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

        self.replay_memory = deque(maxlen=REPLAY_SIZE)
        self.target_update_counter = 0

    def _create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), input_shape=STATE_SHAPE, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.1),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.1),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(NUM_ACTIONS)
        ])
        model.compile(optimizer='rmsprop', loss='mse')
        return model
    
    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))

    def get_q_values(self, x):
        return self.model.predict(x)

    def train(self):
        # guarantee the minimum number of samples
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # get current q values and next q values
        # (current_state, action, reward, next_state, done)
        samples = random.sample(self.replay_memory, self.batch_size)
        current_input = np.stack([sample[0] for sample in samples])
        current_q_values = self.model.predict(current_input)
        next_input = np.stack([sample[3] for sample in samples])
        next_q_values = self.target_model.predict(next_input)

        # update q values
        for i, (_, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * np.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        # fit model
        hist = self.model.fit(current_input, current_q_values, batch_size=self.batch_size, verbose=0, shuffle=False)
        loss = hist.history['loss'][0]
        return loss

    def increase_target_update_counter(self):
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def save(self, model_filepath, target_model_filepath):
        self.model.save(model_filepath)
        self.target_model.save(target_model_filepath)

    def load(self, model_filepath, target_model_filepath):
        self.model = keras.models.load_model(model_filepath)
        self.target_model = keras.models.load_model(target_model_filepath)

class DQNTrainer:
    def __init__ (self):
        self.agent = DQNAgent()
        self.env = Car()

        # 하이퍼 파라미터
        self.max_episode = MAX_EPISODE
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.save_freq = 10

        # learning data
        self.rewards_record = []
        self.steps_record = []
    def train(self):
        pbar = tqdm(initial = 0, total = self.max_episode, unit="episodes")
        try:
            for episode in range(self.max_episode):
                time.sleep(3)
                cur_state = self.env.reset()
                measure_time = time.time()
                step, episode_reward, done = 0, 0, False
                print(f"episode : {episode}")
                while not done:
                    if(np.random.randn(1)) <= self.epsilon:
                        action = np.random.randint(NUM_ACTIONS)
                    else:
                        output = self.agent.get_q_values(cur_state)
                        action = np.argmax(output)

                    decision_time = time.time() - measure_time
                    print(len(self.agent.replay_memory))
                    if(decision_time < 0.5):
                        time.sleep(0.5 - decision_time)
                    measure_time = time.time()

                    next_state, reward, done = self.env.step(action)

                    self.agent.update_replay_memory(np.squeeze(cur_state), action, reward, np.squeeze(next_state), done)

                    cur_state = next_state
                    episode_reward += reward
                    step += 1

                    if done:
                        break

                self.agent.stop()

                self.agent.increase_target_update_counter()

                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                if (episode % self.save_freq ) == 0:
                    self.agent.save(f"model{episode}.keras", f"target_model{episode}.keras")

                pbar.update(1)

                self.rewards_record.append(episode_reward)
                self.steps_record.append(step)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            self.agent.save("model.keras", "target_model.keras")
            plt.plot(self.rewards_record, label='reward')
            plt.plot(self.steps_record, label='step')
            plt.show()

learn = DQNTrainer()
learn.train()
