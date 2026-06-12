# Snake AI Learning Project

This repository builds a Snake-playing AI one learning step at a time.

## Environment Setup

Create the project environment:

```text
conda env create -f environment.yml
conda activate snake-ai
```

After `environment.yml` changes, update the existing environment with:

```text
conda env update -f environment.yml --prune
```

## Phase 1: Snake Environment

The current implementation is a dependency-free, configurable Snake game.
Actions are relative to the snake's current direction:

- `0`: turn left
- `1`: continue straight
- `2`: turn right

```python
from snake_ai.game import SnakeGame

game = SnakeGame(width=20, height=20, seed=42)
state = game.reset()

while not state.done:
    state, reward, done, info = game.step(1)
    game.render()
```

Food gives a reward of `10`, collision gives `-10`, and ordinary movement
gives `0`.

Run the tests with:

```text
python -m unittest discover -v
```

## Phase 2: Manual and Random Agents

Play manually in the terminal. Use the arrow keys to choose an absolute
direction, or use `A`/`D` to turn left/right and `W`/Space to continue
straight. Press `Q` to quit.

```text
python -m snake_ai.agents.manual_agent
python -m snake_ai.agents.manual_agent --width 10 --height 10 --seed 42
```

Run the random agent and print per-episode and aggregate score statistics:

```text
python -m snake_ai.agents.random_agent
python -m snake_ai.agents.random_agent --episodes 5 --width 10 --height 10 --render
```

## Phase 3: Real-Time Visualization Dashboard

The Pygame dashboard uses a minimal black game board with a green snake and
red food. It supports manual and random agents, board-size controls, live game
telemetry, pause/single-step execution, and speeds from `1x` through `5000x`.

The game-state panel tracks the maximum snake length reached and the average
final snake length across the last 50 completed episodes.

The lower dashboard chart shows the full current run from episode 1 through
the latest completed episode. It plots the all-time maximum snake length and
the rolling-50 average final snake length.
Current-state Q-action values remain in the upper-right panel, while state
bits, state ID, epsilon, and Q-update calculations appear below them.

```text
python -m snake_ai.visualization.dashboard
python -m snake_ai.visualization.dashboard --mode manual --width 10 --height 10
```

The right-side learning panel currently shows baseline action statistics. Its
snapshot interface is designed for later phases to provide Q-table updates,
neural-network signals, weight/gradient summaries, and backpropagation data.

## Phase 4: Tabular Q-Learning

The Q-learning agent encodes each game situation into 11 binary features:
immediate danger in three relative directions, the current absolute direction,
and the food's relative position. Those features select one row from a
zero-initialized `2048 x 3` Q-table.

Train without opening the GUI:

```text
python -m snake_ai.agents.q_learning --episodes 1000 --seed 42
```

Watch learning happen in the dashboard:

```text
python -m snake_ai.visualization.dashboard --mode q-learning --seed 42
```

The learning panel displays the current state bits and ID, all three Q-values,
epsilon, whether the action explored or exploited, and the complete latest
Q-learning update: old value, reward, target, and new value.

Both Q-learning dashboard modes also show a three-row action-value chart for
the current state. Positive Q-values extend right in green, negative values
extend left in red, the best action has a blue outline, and the most recently
selected action has an amber marker.

### Two-Step Danger Experiment

An additional Q-learning agent expands danger sensing to six relative inputs:

```text
forward-1 | left-1 | right-1 | forward-2 | left-2 | right-2
```

The distance-two inputs indicate whether the cell exactly two spaces away is
currently blocked by a wall or snake body. Combined with direction and food
inputs, this creates a 14-bit state and a `16384 x 3` Q-table.

Train it without the GUI:

```text
python -m snake_ai.agents.q_learning_two_step --episodes 1000 --seed 42
```

Watch it in the dashboard:

```text
python -m snake_ai.visualization.dashboard --mode q-learning-2step --seed 42
```

## Training Metrics And Comparison

Headless command-line training is substantially faster than GUI training
because it does not render frames or process window events. Specify the exact
number of episodes and dump metrics at the end:

```text
python -m snake_ai.agents.q_learning --episodes 10000 --seed 42 --output outputs/stats/q_learning.json
python -m snake_ai.agents.q_learning_two_step --episodes 15000 --seed 42 --output outputs/stats/q_learning_two_step.json
```

The runs may contain different episode counts. Each metrics file preserves its
own episode numbers. Create a comparison figure containing all-time maximum
and rolling-average snake-length plots:

```text
python -m snake_ai.training.compare_stats outputs/stats/q_learning.json outputs/stats/q_learning_two_step.json --output outputs/q_comparison.png
```

The comparison plot uses a rolling average over 500 episodes by default. Use
`--window` to choose a different averaging period.

The dashboard's `Dump Stats` button saves all completed episodes from the
current dashboard run under `outputs/stats/`.
