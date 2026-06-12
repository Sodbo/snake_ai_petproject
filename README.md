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
