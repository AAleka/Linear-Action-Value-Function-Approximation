import gym
import time
import numpy as np
import os


def q_approximation(env, discount=1.0):
    S = env.observation_space.n
    A = env.action_space.n
    Q = np.zeros(S*A)
    Q_w = np.zeros(S*A)

    f = np.zeros((S*A, S*A), int)
    np.fill_diagonal(f, 1)

    w = np.random.rand(S*A, 1)

    p_next = np.random.randint(0, env.action_space.n, env.observation_space.n)
    p_prev = np.copy(p_next)

    alpha = 0.5
    count = 0

    for i in range(1000):
        if count == 0:
            p_prev = np.copy(p_next)

        for s in range(S):
            for a in range(A):
                _, sp, r, _ = env.env.P[s][a][0]
                F = f[s*A+a, :].reshape((1, S * A))
                Q_w[s*A+a] = np.matmul(F, w)

                Q[s*A+a] = r + discount * max(Q[sp*A:sp*A+A])

            p_next[s] = np.argmax(Q_w.reshape((int(Q_w.shape[0] / 6), 6)).T[:, s])

        w -= -0.5*alpha*np.matmul((Q - Q_w), f).reshape((S*A, 1))

        if np.array_equal(p_prev, p_next):
            count += 1

        else:
            count = 0

        if count == 10:
            print(f"Number of iterations until convergence = {i}")
            time.sleep(1)
            break

    return Q_w.reshape((int(Q_w.shape[0] / 6), 6)).T


env = gym.make('Taxi-v3')
observation = env.reset()

Q = q_approximation(env)

done = False
total_reward = 0

for i in range(3):
    done = False
    total_reward = 0
    while not done:
        os.system('cls')
        env.render()
        action = np.argmax(Q[:, env.env.s])
        observation, reward, done, info = env.step(action)
        total_reward += reward
        time.sleep(0.5)
        if done:
            observation = env.reset()
            print('Done with reward:', total_reward)
            time.sleep(1)
env.close()
