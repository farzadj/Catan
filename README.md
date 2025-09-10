# Catan

Catan is a self-contained Catan engine with a simple GUI board editor and a bot player able to use an MCTS-style search. This repo contains only the minimal files needed to run the engine and GUI:

- `catan_plus.py`: Core game engine and a CLI for self-play.
- `gui_image_sim.py`: Tkinter GUI board editor and single-game runner.

## Features

- Core rules: settlements, cities, roads, dice production, robber, ports, dev cards (knight, year of plenty, monopoly, road building, VP), largest army, longest road.
- Bot play: heuristic baseline with optional MCTS-style search (beam width, depth) for stronger decisions.
- GUI editor: build custom boards (resources, tokens, ports), place pieces, and run a game to completion.
- CLI self-play: simulate one or many games and report wins/turn counts.

## Requirements

- Python 3.10+
- Tkinter (bundled with standard Python on Windows/macOS; on some Linux distros install `python3-tk`).
- Optional: `ttkbootstrap` for a nicer dark theme in the GUI: `pip install ttkbootstrap`.

No other third-party packages are required.

## Quick Start

Clone or download the repo, then from the repo folder run either the CLI or the GUI.

### CLI (engine only)

Run a game with default settings:

```
python catan_plus.py
```

Common options:

- `--seed <int>`: RNG seed (default 42)
- `--vp <int>`: target victory points (default 10)
- `--turns <int>`: max turns before stopping (default 300)
- `--players <2-4>`: number of players (default 2)
- `--mcts`: enable stronger MCTS bot for player 0
- `--mcts-all`: use MCTS for all players
- `--games <N>`: run N games in a row and print win rates
- `--trace <file>`: write a JSON trace of the game

Examples:

- One game, default bots:
  
  ```
  python catan_plus.py --seed 123 --vp 10 --turns 300 --players 3
  ```

- One game with MCTS bot for player 0:
  
  ```
  python catan_plus.py --mcts
  ```

- Ten games, MCTS for all players, report win rates:
  
  ```
  python catan_plus.py --games 10 --mcts-all
  ```

CLI outputs:

- `catan_plus_log.txt`: human-readable log of actions
- `trace.json` (or `--trace` path): full machine trace for analysis

### GUI (board editor + single game)

Launch the GUI:

```
python gui_image_sim.py
```

Key controls:

- Players: set 2–4 players and optionally rename each.
- Strong Bots: toggle stronger MCTS-style bot decisions.
- Beam / Depth: tune the MCTS beam width and search depth.
- Random Map: auto-randomize resources/tokens/ports to a valid setup.
- Save / Open: save or load a board JSON.
- Run One Game: runs a single full game and prints a summary + log.

Notes:

- The GUI’s “Run Simulation” (multi-game batch) references a helper not included in this minimal bundle. For repeated simulations, use the CLI with `--games N`.
- If `ttkbootstrap` is installed, a dark theme is applied automatically; otherwise the default Tk theme is used.

## Project Layout

This pared-down repo intentionally includes only two files to keep usage simple:

- `catan_plus.py`: full rules + bots + CLI
- `gui_image_sim.py`: editor and visual runner (Tkinter)

## License

This repository is provided as-is by the project author. Add a license here if you intend to open-source.
