Updated step-by-step plan:

## SNAKE AI learning project

### Phase 0 — Project setup

Goal: create clean project skeleton.

```textd
snake_ai/
├── game/
├── agents/
├── training/
├── visualization/
├── outputs/
└── README.md
```

No AI yet.

---

## Phase 1 — Snake environment

Goal: build the game itself.

Requirements:

* configurable board size
* default: `20 × 20`
* allow custom sizes: `8×8`, `10×10`, `30×30`, `50×50`
* no hardcoded board dimensions
* actions:

  * `0 = left`
  * `1 = straight`
  * `2 = right`
* API:

```python
game = SnakeGame(width=20, height=20)

state = game.reset()
next_state, reward, done, info = game.step(action)
game.render()
```

---

## Phase 2 — Manual/random agent

Goal: test the game before learning.

Implement:

* manual keyboard play
* random agent
* basic score tracking
* simple rendering

This checks that the environment works.

---

## Phase 3 — Real-time visualization dashboard

Goal: visualize everything while the agent plays.

Dashboard must show:

* Snake field
* board size controls
* episode
* step
* score
* reward
* selected action
* current state
* speed buttons:

```text
Pause | Step | 1x | 5x | 10x | 50x | 100x | 500x | Reset
```

This dashboard should exist before Q-learning.

---

## Phase 4 — Tabular Q-learning

Goal: learn without neural networks.

Implement:

* discrete state representation
* Q-table
* epsilon-greedy policy
* Q-learning update

Visualize live:

* current state ID
* `Q(left)`
* `Q(straight)`
* `Q(right)`
* old Q-value
* reward
* target
* new Q-value

---

## Phase 5 — Manual neural network with NumPy

Goal: understand neural networks from scratch.

Architecture:

```text
11 inputs → 64 → 64 → 3 outputs
```

Implement manually:

* weights
* biases
* forward pass
* ReLU
* MSE loss
* backpropagation
* SGD

Visualize:

* input state
* activations
* Q-values
* loss
* gradients
* weight update summaries

---

## Phase 6 — Manual NumPy DQN

Goal: replace Q-table with neural network.

Implement:

* replay buffer
* mini-batch training
* target Q-value
* epsilon decay
* checkpoints

Visualize:

* replay buffer size
* sampled batch
* Q-values
* loss curve
* score curve

---

## Phase 7 — PyTorch DQN

Goal: reproduce the same DQN in PyTorch.

Implement:

* `torch.nn.Module`
* `torch.optim`
* CUDA support
* same Snake environment
* same dashboard interface

---

## Phase 8 — Checkpoints and videos

Save snapshots at:

```text
episode 0
episode 10
episode 100
episode 1000
```

For each:

* save model
* record GIF/MP4
* save metrics

---

## Phase 9 — Comparison

Compare:

* random agent
* tabular Q-learning
* manual NumPy DQN
* PyTorch DQN

Metrics:

* average score
* max score
* learning speed
* stability
* behavior quality
* effect of board size

---

Best development strategy:

```text
One phase = one Codex prompt.
You review and run each phase before moving on.
```

Start with **Phase 0 + Phase 1 only**.
