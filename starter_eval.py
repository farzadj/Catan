import json
import argparse
import os, sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# Allow running as a script from repo root
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from catan_plus import Game, Player, Board, Hex, add_edge


def build_board(board_json: Dict) -> Board:
    # Reconstruct Board from viewer-style JSON
    nodes = {int(k): tuple(v) for k, v in board_json["nodes"].items()}
    hexes = {}
    for hid_str, h in board_json["hexes"].items():
        hid = int(hid_str)
        # q,r,center are not used in gameplay; default to zeros
        hexes[hid] = Hex(id=hid, q=0, r=0, center=(0.0, 0.0), resource=h.get("resource"), number=h.get("number"), nodes=h.get("nodes", []))
    edges = set()
    for u, v in board_json.get("edges", []):
        edges.add(add_edge((int(u), int(v))))
    node_to_hexes = defaultdict(list)
    for hid, h in hexes.items():
        for nid in h.nodes:
            node_to_hexes[nid].append(hid)
    robber_hex = int(board_json.get("robber_hex", 0))
    ports = {int(k): v for k, v in board_json.get("ports", {}).items()}
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    return Board(hexes=hexes, nodes=nodes, edges=edges, node_to_hexes=node_to_hexes, robber_hex=robber_hex, ports=ports, adj=dict(adj))


def build_players(players_json: List[Dict]) -> List[Player]:
    players: List[Player] = []
    for pid, pj in enumerate(players_json):
        p = Player(id=pid, name=pj.get("name", f"P{pid}"))
        p.settlements = set(int(n) for n in pj.get("settlements", []))
        p.cities = set(int(n) for n in pj.get("cities", []))
        p.roads = set(add_edge((int(u), int(v))) for u, v in pj.get("roads", []))
        p.hand = Counter({k: int(v) for k, v in pj.get("hand", {}).items()})
        # VP from board pieces
        p.vp = len(p.settlements) + 2 * len(p.cities)
        # piece stock
        p.roads_left = max(0, 15 - len(p.roads))
        p.settlements_left = max(0, 5 - len(p.settlements))
        p.cities_left = max(0, 4 - len(p.cities))
        players.append(p)
    return players


def simulate(init: Dict, games: int, max_turns: int, use_mcts: bool=False, mcts_all: bool=False,
             mcts_beam_width: int = 6, mcts_max_depth: int = 3):
    board = build_board(init["board"])
    players_json = init["players"]
    if not (2 <= len(players_json) <= 4):
        raise SystemExit("starter_eval supports 2–4 players.")

    wins = [0 for _ in players_json]
    turns_to_win: List[int] = []
    for k in range(games):
        seed = int(init.get("seed", 42)) + k
        g = Game(seed=seed, target_vp=int(init.get("target_vp", 10)), max_turns=max_turns,
                 use_mcts_bot=use_mcts, num_players=len(players_json), mcts_all=mcts_all,
                 mcts_beam_width=int(mcts_beam_width), mcts_max_depth=int(mcts_max_depth))
        # Override board and players
        g.board = board
        g.players = build_players(players_json)
        # Rebuild bot map to match players
        # Rebuild simple bots for each player
        from catan_plus import HeuristicBot, MCTSBot
        g.bots = {p.id: (MCTSBot(p.id, p.name) if (use_mcts and p.id==0) else HeuristicBot(p.id, p.name)) for p in g.players}
        # Start from current state without setup placements
        g.snapshot("Start from given state")
        g.play_loop()
        # Determine winner
        best = max(range(len(g.players)), key=lambda i: (g.players[i].vp + g.players[i].hidden_vp, sum(g.players[i].hand.values()), -i))
        wins[best] += 1
        turns_to_win.append(g.turn)
    return wins, turns_to_win


def main():
    ap = argparse.ArgumentParser(description="Estimate win probabilities from a given initial Catan state (2-4 players)")
    ap.add_argument("--init", required=True, help="Path to JSON with board + players placements")
    ap.add_argument("--games", type=int, default=500, help="# simulations to run")
    ap.add_argument("--turns", type=int, default=300, help="Max turns per game")
    ap.add_argument("--mcts", action="store_true", help="Use MCTS bot for player 0")
    ap.add_argument("--mcts-all", action="store_true", help="Use MCTS bot for all players")
    ap.add_argument("--out", type=str, default="results.json", help="Where to write summary JSON")
    ap.add_argument("--beam", type=int, default=6, help="MCTS beam width (if MCTS enabled)")
    ap.add_argument("--depth", type=int, default=3, help="MCTS max depth (if MCTS enabled)")
    args = ap.parse_args()

    with open(args.init, "r", encoding="utf-8") as f:
        init = json.load(f)
    wins, turns = simulate(init, args.games, args.turns, use_mcts=args.mcts, mcts_all=args.mcts_all,
                           mcts_beam_width=args.beam, mcts_max_depth=args.depth)
    names = [p.get("name", f"P{i}") for i, p in enumerate(init["players"])]
    total = sum(wins) if sum(wins) > 0 else 1
    results = {
        "players": [
            {"id": i, "name": names[i], "wins": wins[i], "win_rate": wins[i]/total}
            for i in range(len(names))
        ],
        "games": args.games,
        "avg_turns": sum(turns)/len(turns) if turns else None,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
