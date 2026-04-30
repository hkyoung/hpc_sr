from doctest import testfile
import numpy as np
import matplotlib.pyplot as plt

class LinearTrack:
    def __init__(self, n, p_right):
        self.n = n
        self.p_right = p_right
        self.adj = np.zeros((n, n))
        i = np.arange(n-1)
        self.adj[i, i+1] = 1
        self.adj[i+1, i] = 1
        self.trans = p_right * np.triu(self.adj) + (1 - p_right) * np.tril(self.adj)
        self.trans = self.trans / self.trans.sum(axis=1)

class Ring:
    def __init__(self, n, p_right):
        self.n = n
        self.p_right = p_right
        self.adj = np.zeros((n, n))
        i = np.arange(n)
        self.adj[i, (i+1)%n] = 1
        self.adj[(i+1)%n, i] = 1
        self.trans = np.zeros((n, n))
        self.trans[i, (i+1)%n] = p_right
        self.trans[(i+1)%n, i] = 1 - p_right
        self.trans = self.trans / self.trans.sum(axis=1)

def compute_sr(trans, gamma):
    return np.linalg.inv(np.eye(trans.shape[0]) - gamma * trans)

def compute_softmax_policy(adj, value, beta):
    policy = np.zeros(adj.shape)
    for state in range(adj.shape[0]):
        next_states = np.where(adj[state] > 0)[0]
        if len(next_states) == 0:
            continue
        value_next = value[next_states]
        value_next = value_next - value_next.max()
        weights = np.exp(beta * value_next)
        probs = weights / weights.sum()
        policy[state, next_states] = probs
    return policy

def sr_td_error(M, s0, s1, gamma):
    e = np.zeros(M.shape[0])
    e[s0] = 1
    return e + gamma * M[s1] - M[s0]

def sample_trajectory(T, start_state, n_steps=30):
    traj = [start_state]
    curr_state = start_state
    for _ in range(n_steps):
        curr_state = np.random.choice(np.arange(T.shape[0]), p=T[curr_state])
        traj.append(curr_state)
    return traj

def plot_sr_row(M, s, reward_loc=None, figsize=(7, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(M[s], '.-')
    if reward_loc:
        ax.axvline(reward_loc, color='r', linestyle='--')
    ax.set_xlabel('Future state')
    ax.set_ylabel('Discounted occupancy')
    ax.set_title(f'SR row at state {s}')
    return fig, ax

def plot_sr_row_comp(Ms, s, reward_loc=None, figsize=(5, 3), labels=None, title=None):
    fig, ax = plt.subplots(figsize=figsize)
    for i, M in enumerate(Ms):
        ax.plot(M[s], '.-', label=labels[i])
    if reward_loc:
        ax.axvline(reward_loc, color='r', linestyle='--')
    if labels:
        ax.legend()
    ax.set_xlabel('Future state')
    ax.set_ylabel('Discounted occupancy')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'SR row at state {s}')
    return fig, ax

def plot_sr_column(M, s, reward_loc=None, figsize=(5, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(M[:, s], '.-')
    if reward_loc:
        ax.axvline(reward_loc, color='r', linestyle='--')
    ax.set_xlabel('Previous state')
    ax.set_ylabel('Discounted occupancy')
    ax.set_title(f'SR column at state {s}')
    return fig, ax

def plot_sr_matrix(M, figsize=(5, 5), vmax=None, title=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(M, cmap='viridis', vmax=vmax)
    ax.set_xlabel('State j')
    ax.set_ylabel('State i')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('SR matrix')
    return fig, ax

def plot_teleport_asymmetry(results, figsize=(5, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    for dist in np.unique(results[:,1]):
        mask = results[:,1] == dist
        ax.plot(results[mask,0], results[mask,4], '.-', label=f'd={dist}')
    ax.axhline(0, linestyle='--', color='k', lw=1)
    ax.set_xlabel('Start state')
    ax.set_ylabel('Backward TD error - Forward TD error')
    ax.set_title('Teleport asymmetry across start states')
    ax.legend()
    return fig, ax

def plot_teleport_asymmetry_bar(results, figsize=(5, 3), title=None):
    fig, ax = plt.subplots(figsize=figsize)
    for i_dist, dist in enumerate(np.unique(results[:,1])):
        mask = results[:,1] == dist
        ax.bar(i_dist, np.mean(results[mask,4]))
    ax.axhline(0, linestyle='--', color='k', lw=1)
    ax.set_xlabel('Teleport distance')
    ax.set_ylabel('Backward - Forward TD error')
    ax.set_xticks(np.arange(len(np.unique(results[:,1]))), np.unique(results[:,1]).astype(int))
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Teleport asymmetry across teleport distances')
    return fig, ax

