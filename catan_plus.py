
import math, random, json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

RESOURCES = ["brick", "wood", "wool", "grain", "ore"]
BUILD_COST = {
    "road": Counter({"brick": 1, "wood": 1}),
    "settlement": Counter({"brick": 1, "wood": 1, "wool": 1, "grain": 1}),
    "city": Counter({"ore": 3, "grain": 2}),
    "dev": Counter({"wool":1, "grain":1, "ore":1}),
}
DICE_WEIGHTS = {2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1}
DEV_DECK_DEFAULT = (["knight"]*14 + ["year_of_plenty"]*2 + ["monopoly"]*2 + ["road_building"]*2 + ["vp"]*5)

@dataclass
class Hex:
    id: int
    q: int
    r: int
    center: Tuple[float, float]
    resource: Optional[str] = None
    number: Optional[int] = None
    nodes: List[int] = field(default_factory=list)

@dataclass
class Board:
    hexes: Dict[int, Hex]
    nodes: Dict[int, Tuple[float, float]]
    edges: Set[Tuple[int,int]]
    node_to_hexes: Dict[int, List[int]]
    robber_hex: int
    ports: Dict[int, Dict] = field(default_factory=dict)
    adj: Dict[int, Set[int]] = field(default_factory=dict)

@dataclass
class Player:
    id: int
    name: str
    vp: int = 0
    hidden_vp: int = 0
    settlements: Set[int] = field(default_factory=set)
    cities: Set[int] = field(default_factory=set)
    roads: Set[Tuple[int,int]] = field(default_factory=set)
    hand: Counter = field(default_factory=Counter)
    dev_hand: Counter = field(default_factory=Counter)
    knights_played: int = 0
    has_largest_army: bool = False
    longest_road_len: int = 0
    has_longest_road: bool = False
    # Piece limits per standard Catan
    roads_left: int = 15
    settlements_left: int = 5
    cities_left: int = 4
    # Turn-scoped dev card restrictions
    devs_bought_this_turn: Counter = field(default_factory=Counter)
    dev_played_this_turn: bool = False

def axial_to_pixel(q, r, size=1.0):
    x = size * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
    y = size * (1.5 * r)
    return (x, y)

def hex_corners(center, size=1.0):
    cx, cy = center
    corners = []
    for k in range(6):
        angle = math.radians(60 * k + 30)
        x = cx + size * math.cos(angle)
        y = cy + size * math.sin(angle)
        corners.append((x, y))
    return corners

def round_key(pt, scale=1000):
    return (int(round(pt[0]*scale)), int(round(pt[1]*scale)))

def add_edge(e):
    u,v = e
    return (u,v) if u < v else (v,u)

def adjacent_nodes(node, edges):
    adj = set()
    for u,v in edges:
        if u == node: adj.add(v)
        elif v == node: adj.add(u)
    return adj

def node_neighbors(node, edges):
    return adjacent_nodes(node, edges)

def can_afford(hand: Counter, cost: Counter) -> bool:
    for res, cnt in cost.items():
        if hand[res] < cnt:
            return False
    return True

def pay_cost(hand: Counter, cost: Counter):
    for res, cnt in cost.items():
        hand[res] -= cnt
        if hand[res] == 0:
            del hand[res]

def make_standard_board(seed=0, rng: Optional[random.Random]=None) -> Board:
    rng = rng or random.Random(seed)
    hex_coords = []
    radius = 2
    for q in range(-radius, radius+1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2+1):
            hex_coords.append((q, r))

    hexes = {}
    for i, (q, r) in enumerate(hex_coords):
        center = axial_to_pixel(q, r, size=1.0)
        hexes[i] = Hex(id=i, q=q, r=r, center=center)

    node_id_by_pt = {}
    nodes = {}
    edges = set()
    node_to_hexes = defaultdict(list)

    next_node_id = 0
    for hid, h in hexes.items():
        corners = hex_corners(h.center, size=1.0)
        hex_node_ids = []
        for c in corners:
            key = round_key(c, scale=1000)
            if key not in node_id_by_pt:
                node_id_by_pt[key] = next_node_id
                nodes[next_node_id] = c
                next_node_id += 1
            nid = node_id_by_pt[key]
            hex_node_ids.append(nid)
            node_to_hexes[nid].append(hid)

        for i in range(6):
            u = hex_node_ids[i]
            v = hex_node_ids[(i+1)%6]
            edges.add(add_edge((u, v)))

        h.nodes = hex_node_ids

    # Assign resources
    resource_pool = (["desert"]*1 + ["brick"]*3 + ["wood"]*4 + ["wool"]*4 + ["grain"]*4 + ["ore"]*3)
    rng.shuffle(resource_pool)
    robber_hex = None
    for hid, h in hexes.items():
        res = resource_pool.pop()
        h.resource = None if res == "desert" else res
        if res == "desert":
            h.number = None
            robber_hex = hid
    if robber_hex is None:
        robber_hex = 0

    # Prepare hex adjacency (share an edge => neighbors)
    hex_neighbors = defaultdict(set)
    node_to_hexset = defaultdict(set)
    for hid, h in hexes.items():
        for nid in h.nodes:
            node_to_hexset[nid].add(hid)
    # Two hexes share an edge if they share at least 2 nodes
    for nid, hids in node_to_hexset.items():
        for a in hids:
            for b in hids:
                if a >= b:
                    continue
                # Count shared nodes between hex a and b
                shared = len(set(hexes[a].nodes) & set(hexes[b].nodes))
                if shared >= 2:
                    hex_neighbors[a].add(b)
                    hex_neighbors[b].add(a)

    # Assign numbers in official spiral order, skipping desert.
    # Spiral path: ring radius=2 outer → inner → center, starting from top and CCW.
    def ring_coords(radius):
        if radius == 0:
            return [(0,0)]
        results = []
        # start at (0, -radius)
        q, r = 0, -radius
        directions = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]  # axial CCW
        for dir_q, dir_r in directions:
            for _ in range(radius):
                results.append((q, r))
                q += dir_q; r += dir_r
        return results
    spiral = ring_coords(2) + ring_coords(1) + ring_coords(0)
    # Map coord -> hex id
    coord_to_hid = {(h.q, h.r): hid for hid,h in hexes.items()}
    order_hids = [coord_to_hid[(q,r)] for (q,r) in spiral if (q,r) in coord_to_hid]
    # Official token order (18 tokens):
    token_order = [5,2,6,3,8,10,9,12,11,4,8,10,9,4,5,6,3,11]
    ti = 0
    for hid in order_hids:
        if hexes[hid].resource is None:
            continue
        hexes[hid].number = token_order[ti]
        ti += 1

    # Build adjacency map once
    adj = defaultdict(set)
    for u,v in edges:
        adj[u].add(v); adj[v].add(u)
    board = Board(hexes=hexes, nodes=nodes, edges=edges, node_to_hexes=node_to_hexes, robber_hex=robber_hex, ports={}, adj=dict(adj))

    # Ports: choose 9 perimeter nodes evenly spaced around the board and
    # assign 4 generic (3:1) and 5 specific (2:1) alternating for spacing.
    border = [nid for nid in nodes if len(board.adj.get(nid, ())) <= 2]
    # order by angle around centroid
    if border:
        cx = sum(nodes[n][0] for n in border) / len(border)
        cy = sum(nodes[n][1] for n in border) / len(border)
        border.sort(key=lambda n: math.atan2(nodes[n][1]-cy, nodes[n][0]-cx))
    step = max(1, int(round(len(border)/9))) if border else 1
    chosen = []
    used = set()
    idx = 0
    while len(chosen) < 9 and border:
        n = border[idx % len(border)]
        if n not in used:
            chosen.append(n); used.add(n)
        idx += step
        if idx > len(border)*3:
            break
    while len(chosen) < 9 and border:
        for n in border:
            if n not in used:
                chosen.append(n); used.add(n)
                if len(chosen) == 9:
                    break
            
    ports = {}
    specific_resources = ["brick","wood","wool","grain","ore"]
    rng.shuffle(specific_resources)
    # Alternate generic/specific starting with generic
    specifics = 0; generics = 0
    for k, nid in enumerate(chosen):
        if specifics < 5 and (k % 2 == 1 or generics >= 4):
            r = specific_resources[specifics]
            ports[nid] = {"type":"resource", "resource":r, "rate":2}
            specifics += 1
        else:
            ports[nid] = {"type":"generic", "resource":None, "rate":3}
            generics += 1

    board.ports = ports
    return board

def compute_longest_road_length(board: Board, player: Player, opponents: List[Player]) -> int:
    player_edges = set(player.roads)
    if not player_edges:
        return 0
    blocked_nodes = set()
    for opp in opponents:
        blocked_nodes |= opp.settlements
        blocked_nodes |= opp.cities

    adj2 = defaultdict(set)
    for u,v in player_edges:
        if u in blocked_nodes or v in blocked_nodes: 
            continue
        adj2[u].add(v); adj2[v].add(u)

    if not adj2:
        return 0

    best = 0
    def dfs(curr, used_edges, length):
        nonlocal best
        best = max(best, length)
        for nxt in list(adj2[curr]):
            e = (min(curr, nxt), max(curr, nxt))
            if e in used_edges: 
                continue
            used_edges.add(e)
            dfs(nxt, used_edges, length+1)
            used_edges.remove(e)

    nodes_considered = set(adj2.keys())
    for start in nodes_considered:
        dfs(start, set(), 0)
    return best

@dataclass
class Game:
    seed: int = 0
    target_vp: int = 10
    max_turns: int = 300
    use_mcts_bot: bool = False
    num_players: int = 2
    mcts_all: bool = False
    # Tunables for MCTS bot
    mcts_beam_width: int = 6
    mcts_max_depth: int = 3

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.board = make_standard_board(seed=self.seed, rng=self.rng)
        self.num_players = max(2, min(4, int(self.num_players)))
        names = ["Bot A", "Bot B", "Bot C", "Bot D"]
        self.players = [Player(i, names[i]) for i in range(self.num_players)]
        self.logs: List[str] = []
        self.dev_deck = DEV_DECK_DEFAULT.copy()
        self.rng.shuffle(self.dev_deck)
        self.current = 0
        self.turn = 0
        self.largest_army_holder: Optional[int] = None
        self.longest_road_holder: Optional[int] = None
        self.trace: List[Dict] = []
        self.bots = {
            i: (MCTSBot(i, names[i]) if (self.use_mcts_bot and (self.mcts_all or i == 0)) else HeuristicBot(i, names[i]))
            for i in range(self.num_players)
        }

    def log(self, msg): self.logs.append(msg)

    def log_full_state(self, tag: str):
        """Append a detailed JSON snapshot of the current game state to the text log.
        Includes per-player hands, devs, piece positions, stocks, board ports, and robber.
        """
        state = {
            "turn": self.turn,
            "current": self.current,
            "tag": tag,
            "largest_army_holder": self.largest_army_holder,
            "longest_road_holder": self.longest_road_holder,
            "board": {
                "robber_hex": self.board.robber_hex,
                "ports": self.board.ports,
                "nodes": {int(k): (float(v[0]), float(v[1])) for k, v in self.board.nodes.items()},
            },
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "vp": p.vp,
                    "hidden_vp": p.hidden_vp,
                    "hand": dict(p.hand),
                    "dev_hand": dict(p.dev_hand),
                    "settlements": sorted(p.settlements),
                    "cities": sorted(p.cities),
                    "roads": [list(e) for e in sorted(p.roads)],
                    "roads_left": p.roads_left,
                    "settlements_left": p.settlements_left,
                    "cities_left": p.cities_left,
                    "knights_played": p.knights_played,
                    "has_largest_army": p.has_largest_army,
                    "longest_road_len": p.longest_road_len,
                    "has_longest_road": p.has_longest_road,
                }
                for p in self.players
            ],
        }
        try:
            self.log("STATE " + tag + ":\n" + json.dumps(state, indent=2))
        except Exception:
            # Fallback: minimal serialization
            self.log("STATE " + tag + ": " + str(state))

    def snapshot(self, note=""):
        snap = {
            "turn": self.turn,
            "current": self.current,
            "note": note,
            "board": {
                "robber_hex": self.board.robber_hex,
                "hexes": {hid: {"resource":h.resource, "number":h.number, "nodes":h.nodes} for hid,h in self.board.hexes.items()},
                "nodes": self.board.nodes,
                "edges": list(self.board.edges),
                "ports": self.board.ports,
            },
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "vp": p.vp,
                    "hidden_vp": p.hidden_vp,
                    "settlements": list(p.settlements),
                    "cities": list(p.cities),
                    "roads": [list(e) for e in p.roads],
                    "hand": dict(p.hand),
                    "dev_hand": dict(p.dev_hand),
                    "knights_played": p.knights_played,
                    "has_largest_army": p.has_largest_army,
                    "longest_road_len": p.longest_road_len,
                    "has_longest_road": p.has_longest_road,
                } for p in self.players
            ]
        }
        self.trace.append(snap)

    def _deficit(self, hand: Counter, cost: Counter) -> int:
        return sum(max(0, cost[r] - hand[r]) for r in cost)

    def _best_build_deficit(self, hand: Counter) -> int:
        """Return the best (lowest) deficit over main build targets for a given hand."""
        return min(
            self._deficit(hand, BUILD_COST["city"]),
            self._deficit(hand, BUILD_COST["settlement"]),
            self._deficit(hand, BUILD_COST["road"]),
            self._deficit(hand, BUILD_COST["dev"])
        )

    def p2p_trade(self, proposer: Player, responder: Player) -> bool:
        """Smarter P2P trading: allow variable ratios and ensure mutual benefit.

        - Proposer selects a wanted resource based on highest current deficit across
          city/settlement first, but considers all four build types.
        - Considers offers of k:1 (k in 1..4) based on each side's bank/port rate so
          responder never does worse than their own bank/port alternative.
        - Accepts only if both players' best-build deficit improves (strictly for the proposer,
          non-worse-and-usually-better for responder), preferring Pareto-improving trades.
        """
        # Early exits
        if not responder.hand:
            return False

        # Determine proposer's needs ordered by severity
        def deficits_by_resource(hand: Counter):
            want_map = defaultdict(int)
            for cost in (BUILD_COST["city"], BUILD_COST["settlement"], BUILD_COST["dev"], BUILD_COST["road"]):
                for r in cost:
                    need = max(0, cost[r] - hand[r])
                    if need > 0:
                        want_map[r] = max(want_map[r], need)
            # sort by need desc
            return [r for r,_ in sorted(want_map.items(), key=lambda kv: kv[1], reverse=True)]

        wants = deficits_by_resource(proposer.hand)
        if not wants:
            return False

        best_choice = None  # (score, give, want, k)
        prop_before = self._best_build_deficit(proposer.hand)
        resp_before = self._best_build_deficit(responder.hand)

        # Candidate gives: anything proposer holds, excluding the want
        give_candidates = [r for r,c in proposer.hand.items() if c > 0]

        for want in wants:
            if responder.hand[want] <= 0:
                continue
            for give in give_candidates:
                if give == want:
                    continue
                # Maximum reasonable k to consider
                for k in (1, 2, 3, 4):
                    if proposer.hand[give] < k:
                        continue
                    # Ensure responder gets at least their bank/port alternative value
                    required_k = self.best_trade_rate(responder, want)
                    if k < required_k:
                        continue
                    # Simulate effects
                    ph = proposer.hand.copy()
                    rh = responder.hand.copy()
                    # Apply trade
                    ph[give] -= k
                    if ph[give] == 0:
                        del ph[give]
                    ph[want] += 1
                    rh[want] -= 1
                    if rh[want] == 0:
                        del rh[want]
                    rh[give] += k

                    prop_after = self._best_build_deficit(ph)
                    resp_after = self._best_build_deficit(rh)

                    prop_gain = prop_before - prop_after
                    resp_gain = resp_before - resp_after

                    # Proposer must strictly improve. Responder should not be worse
                    # than their baseline bank move of converting 1 want at their rate.
                    if prop_gain <= 0:
                        continue

                    # Also require that responder is not worse off than bank/port conversion
                    # approximate: simulate responder's best-rate bank trade of 1 want into 'give'
                    # If k >= required_k, typically non-worse. Still, avoid harming responder grossly.
                    if resp_after > resp_before:
                        continue

                    # Score: prioritize mutual improvement, then smaller k (cheaper for proposer)
                    score = 10 * prop_gain + 6 * resp_gain - 0.1 * k
                    if best_choice is None or score > best_choice[0]:
                        best_choice = (score, give, want, k)

        if best_choice is None:
            return False

        _, give, want, k = best_choice
        # Execute the chosen trade
        proposer.hand[give] -= k
        if proposer.hand[give] == 0:
            del proposer.hand[give]
        proposer.hand[want] += 1
        responder.hand[want] -= 1
        if responder.hand[want] == 0:
            del responder.hand[want]
        responder.hand[give] += k
        ratio_txt = f"{k}:1"
        self.log(f"{proposer.name} trades {ratio_txt} P2P with {responder.name}: gives {k} {give} for 1 {want}.")
        self.snapshot(f"P2P {ratio_txt} {proposer.name}:{give}->{want}")
        return True

    def all_roads(self) -> Set[Tuple[int,int]]:
        roads = set()
        for p in self.players: roads |= p.roads
        return roads

    def best_trade_rate(self, player: Player, give_res: str) -> int:
        rate = 4
        port_nodes = set(self.board.ports.keys())
        owned_nodes = player.settlements | player.cities
        owned_ports = [self.board.ports[n] for n in owned_nodes if n in port_nodes]
        has_generic = any(p["type"]=="generic" for p in owned_ports)
        has_specific = any(p["type"]=="resource" and p["resource"]==give_res for p in owned_ports)
        if has_specific: rate = 2
        elif has_generic: rate = 3
        return rate

    def setup_initial_placements(self):
        order = list(range(self.num_players)) + list(range(self.num_players-1, -1, -1))
        for i, pid in enumerate(order):
            player = self.players[pid]
            others = [p for p in self.players if p.id != pid]
            nid = self.bots[pid].choose_initial_settlement(self, player, others)
            while True:
                ok = True
                for p in self.players:
                    for n in p.settlements:
                        if nid == n or nid in self.board.adj.get(n, ()): 
                            ok = False; break
                    if not ok: break
                if ok: break
                legal = self.legal_settlement_spots(player, others, require_connection=False)
                if nid in legal:
                    legal.remove(nid)
                if not legal:
                    candidates = [x for x in self.board.nodes if all(
                        x != n and x not in self.board.adj.get(n, ()) for p in self.players for n in p.settlements
                    )]
                    nid = self.rng.choice(candidates) if candidates else nid
                    break
                nid = max(legal, key=lambda n: self.node_expectation(n))

            player.settlements.add(nid); player.vp += 1; player.settlements_left -= 1
            e = self.bots[pid].choose_initial_road(self, player, others, from_node=nid)
            if e in self.all_roads():
                nbs = list(self.board.adj.get(nid, ()))
                e = add_edge((nid, self.rng.choice(nbs))) if nbs else add_edge(self.rng.choice(list(self.board.edges)))
            player.roads.add(e); player.roads_left -= 1
            self.log(f"{player.name} places initial settlement at node {nid} and road {e}. VP={player.vp}")
            self.snapshot(f"Initial placement {player.name}")

            if i >= self.num_players:
                gained = Counter()
                for hid in self.board.node_to_hexes[nid]:
                    h = self.board.hexes[hid]
                    if h.resource:
                        gained[h.resource] += 1
                        player.hand[h.resource] += 1
                if sum(gained.values()) > 0:
                    self.log(f"{player.name} gains starting resources: {dict(gained)}.")
                    self.snapshot(f"{player.name} starting resources {dict(gained)}")
        # Log comprehensive state after setup
        self.log_full_state("after_setup")

    def node_expectation(self, nid: int) -> float:
        # Enhanced node scoring: base pip weight + resource diversity bonus + ore/wheat synergy + port bonus
        pip = 0.0
        resources = []
        for hid in self.board.node_to_hexes[nid]:
            h = self.board.hexes[hid]
            if h.number is not None and hid != self.board.robber_hex:
                pip += DICE_WEIGHTS.get(h.number, 0)
            if h.resource:
                resources.append(h.resource)
        distinct = len(set(resources))
        diversity_bonus = 0.35 * distinct
        ow = (resources.count("ore")>0) + (resources.count("grain")>0)
        ow_bonus = 0.6 if ow==2 else (0.2 if ow==1 else 0.0)
        port_bonus = 0.0
        if nid in self.board.ports:
            p = self.board.ports[nid]
            port_bonus = 0.8 if p.get("type")=="generic" else 0.6
        return pip + diversity_bonus + ow_bonus + port_bonus

    def legal_settlement_spots(self, player: Player, others: List[Player], require_connection=True) -> List[int]:
        occupied = set()
        for p in [player] + others:
            occupied |= p.settlements; occupied |= p.cities

        banned = set()
        for n in occupied:
            banned.add(n)
            for m in self.board.adj.get(n, ()): 
                banned.add(m)

        legal = []
        connected_nodes = self.player_connected_nodes(player)
        for nid in self.board.nodes.keys():
            if nid in banned: continue
            if require_connection:
                # Strict rule: settlement must be at the end of one of your roads
                if not any((u == nid or v == nid) for (u,v) in player.roads):
                    continue
            legal.append(nid)
        return legal

    def player_connected_nodes(self, player: Player) -> Set[int]:
        owned_nodes = set(player.settlements) | set(player.cities)
        # Nodes blocked by opponents' settlements/cities — cannot traverse THROUGH them
        opponent_occupied = set()
        for p in self.players:
            if p.id == player.id:
                continue
            opponent_occupied |= set(p.settlements)
            opponent_occupied |= set(p.cities)

        adj = defaultdict(set)
        for u, v in player.roads:
            adj[u].add(v)
            adj[v].add(u)

        seen = set()
        stack = list(owned_nodes) + [n for e in player.roads for n in e]
        for start in stack:
            if start in seen:
                continue
            dq = deque([start])
            seen.add(start)
            while dq:
                x = dq.popleft()
                # Do not expand neighbors past an opponent-occupied node (unless it's our own building)
                if x in opponent_occupied and x not in owned_nodes:
                    continue
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        dq.append(y)
        return seen | owned_nodes

    def legal_road_spots(self, player: Player) -> List[Tuple[int,int]]:
        blocked = self.all_roads()
        connected = self.player_connected_nodes(player) | player.settlements | player.cities
        # Nodes that are occupied by an OPPONENT building (settlement/city)
        opponent_occupied = set()
        for p in self.players:
            if p.id == player.id:
                continue
            opponent_occupied |= set(p.settlements)
            opponent_occupied |= set(p.cities)
        legal = []
        for e in self.board.edges:
            if e in blocked:
                continue
            u, v = e
            # Road must attach to our network
            if not (u in connected or v in connected):
                continue
            # Disallow roads that touch an opponent's settlement/city at either endpoint
            # (prevents "building through" or onto an opponent-occupied node)
            if u in opponent_occupied or v in opponent_occupied:
                continue
            legal.append(e)
        return legal

    def distribute_resources(self, roll):
        if roll == 7: return
        for hid, h in self.board.hexes.items():
            if h.number != roll or hid == self.board.robber_hex: 
                continue
            for nid in h.nodes:
                for p in self.players:
                    if nid in p.settlements:
                        if h.resource: p.hand[h.resource] += 1
                    if nid in p.cities:
                        if h.resource: p.hand[h.resource] += 2

    def handle_robber(self, player: Player, others: List[Player]):
        for p in self.players:
            total = sum(p.hand.values())
            if total > 7:
                to_discard = total // 2
                for _ in range(to_discard):
                    choices = list(p.hand.elements())
                    if not choices: break
                    res = self.rng.choice(choices)
                    p.hand[res] -= 1
                    if p.hand[res] == 0: del p.hand[res]
                self.log(f"{p.name} discards {to_discard} cards due to 7.")
                self.snapshot(f"{p.name} discards {to_discard}")
        target = self.bots[player.id].choose_robber_target_hex(self, player, others)
        self.board.robber_hex = target
        h = self.board.hexes[target]
        token_txt = (f"token {h.number}" if h.number is not None else ("desert" if h.resource is None else "no token"))
        self.log(f"{player.name} moves robber to hex {target} ({token_txt}).")
        self.snapshot(f"{player.name} robber to hex {target} ({token_txt})")
        victims = []
        for opp in others:
            if any(nid in opp.settlements or nid in opp.cities for nid in h.nodes):
                if sum(opp.hand.values()) > 0:
                    victims.append(opp)
        if victims:
            target_opp = self.rng.choice(victims)
            stolen = self.rng.choice(list(target_opp.hand.elements()))
            target_opp.hand[stolen] -= 1
            if target_opp.hand[stolen] == 0: del target_opp.hand[stolen]
            player.hand[stolen] += 1
            self.log(f"{player.name} steals 1 {stolen} from {target_opp.name}.")
            self.snapshot(f"{player.name} steals {stolen}")

    def update_special_cards(self):
        lengths = []
        for i, p in enumerate(self.players):
            L = compute_longest_road_length(self.board, p, [op for op in self.players if op.id != p.id])
            p.longest_road_len = L
            lengths.append((L, i))
        cur_holder = self.longest_road_holder
        best_len, best_idx = max(lengths)
        new_holder = None
        if best_len >= 5 and sum(1 for L,i in lengths if L==best_len)==1:
            new_holder = best_idx
        if new_holder != cur_holder:
            if cur_holder is not None:
                prev = self.players[cur_holder]
                if prev.has_longest_road:
                    prev.has_longest_road = False; prev.vp -= 2
                    self.log(f"{prev.name} loses Longest Road."); self.snapshot(f"{prev.name} loses Longest Road")
            if new_holder is not None:
                now = self.players[new_holder]
                if not now.has_longest_road:
                    now.has_longest_road = True; now.vp += 2
                    self.log(f"{now.name} gains Longest Road (+2 VP)."); self.snapshot(f"{now.name} gains Longest Road")
            self.longest_road_holder = new_holder

        counts = [(p.knights_played, i) for i,p in enumerate(self.players)]
        best_knights, idx = max(counts)
        cur = self.largest_army_holder
        new_holder = None
        if best_knights >= 3 and sum(1 for c,i in counts if c==best_knights)==1:
            new_holder = idx
        if new_holder != cur:
            if cur is not None:
                prev = self.players[cur]
                if prev.has_largest_army:
                    prev.has_largest_army = False; prev.vp -= 2
                    self.log(f"{prev.name} loses Largest Army."); self.snapshot(f"{prev.name} loses Largest Army")
            if new_holder is not None:
                now = self.players[new_holder]
                if not now.has_largest_army:
                    now.has_largest_army = True; now.vp += 2
                    self.log(f"{now.name} gains Largest Army (+2 VP)."); self.snapshot(f"{now.name} gains Largest Army")
            self.largest_army_holder = new_holder

    def buy_dev_card(self, player: Player) -> bool:
        if not can_afford(player.hand, BUILD_COST["dev"]) or not self.dev_deck:
            return False
        pay_cost(player.hand, BUILD_COST["dev"])
        card = self.dev_deck.pop()
        player.dev_hand[card] += 1
        player.devs_bought_this_turn[card] += 1
        self.log(f"{player.name} buys a development card."); self.snapshot(f"{player.name} buys dev card")
        if card == "vp":
            # VP cards remain hidden but count toward victory
            player.hidden_vp += 1
        return True

    def _can_play_dev(self, player: Player, card: str) -> bool:
        # Only one dev card per turn; cannot play a card bought this turn
        if player.dev_played_this_turn:
            return False
        if player.dev_hand[card] - player.devs_bought_this_turn[card] <= 0:
            return False
        return True

    def play_knight(self, player: Player, others: List[Player]) -> bool:
        if not self._can_play_dev(player, "knight"):
            return False
        player.dev_hand["knight"] -= 1; player.knights_played += 1; player.dev_played_this_turn = True
        self.log(f"{player.name} plays KNIGHT.")
        self.handle_robber(player, others)
        self.update_special_cards()
        return True

    def play_year_of_plenty(self, player: Player, resA: str, resB: str) -> bool:
        if not self._can_play_dev(player, "year_of_plenty"):
            return False
        player.dev_hand["year_of_plenty"] -= 1; player.dev_played_this_turn = True
        player.hand[resA] += 1; player.hand[resB] += 1
        self.log(f"{player.name} plays YEAR OF PLENTY for {resA} and {resB}."); self.snapshot(f"{player.name} YOP {resA},{resB}")
        return True

    def play_monopoly(self, player: Player, others: List[Player], res: str) -> bool:
        if not self._can_play_dev(player, "monopoly"):
            return False
        player.dev_hand["monopoly"] -= 1; player.dev_played_this_turn = True
        total = 0
        for opp in others:
            take = opp.hand[res]
            if take > 0:
                opp.hand[res] -= take; player.hand[res] += take; total += take
        self.log(f"{player.name} plays MONOPOLY on {res}, steals {total} from opponents."); self.snapshot(f"{player.name} Monopoly {res} ({total})")
        return True

    def play_road_building(self, player: Player) -> bool:
        if not self._can_play_dev(player, "road_building"):
            return False
        if player.roads_left <= 0:
            return False
        player.dev_hand["road_building"] -= 1; player.dev_played_this_turn = True
        built = 0
        for _ in range(2):
            if player.roads_left <= 0:
                break
            legal = self.legal_road_spots(player)
            if not legal:
                break
            # Prefer edges that unlock an immediately legal settlement endpoint
            def node_open_for_settlement(nid: int) -> bool:
                for p in self.players:
                    if nid in p.settlements or nid in p.cities:
                        return False
                for m in self.board.adj.get(nid, ()):  # distance rule
                    for p in self.players:
                        if m in p.settlements or m in p.cities:
                            return False
                return True
            def node_threat_level(nid: int) -> int:
                threat = 0
                for opp in self.players:
                    if opp.id == player.id:
                        continue
                    conn = self.player_connected_nodes(opp)
                    if any((u == nid or v == nid) for (u,v) in opp.roads):
                        threat += 2; continue
                    for nb in self.board.adj.get(nid, ()):  # one road away
                        if nb in conn:
                            threat += 1; break
                return threat
            def score_edge(e):
                u,v = e
                u_open = node_open_for_settlement(u); v_open = node_open_for_settlement(v)
                if not (u_open or v_open):
                    def next_best(n):
                        return max([self.node_expectation(x) for x in self.board.adj.get(n, ()) if node_open_for_settlement(x)], default=0)
                    nb = max(next_best(u), next_best(v))
                    thr = min(node_threat_level(u), node_threat_level(v))
                    return -1.0 + 0.5 * nb - 1.0 * thr
                su = self.node_expectation(u) + (2.0 if u_open else 0.0) - 1.2 * node_threat_level(u)
                sv = self.node_expectation(v) + (2.0 if v_open else 0.0) - 1.2 * node_threat_level(v)
                return max(su, sv)
            choice = max(legal, key=score_edge)
            player.roads.add(choice); player.roads_left -= 1
            self.log(f"{player.name} plays ROAD BUILDING and builds road {choice}.")
            self.snapshot(f"{player.name} RoadBuilding {choice}")
            built += 1
        if built == 0:
            return False
        self.update_special_cards()
        return True

    def take_turn(self, pid: int):
        player = self.players[pid]; bot = self.bots[pid]; others = [p for p in self.players if p.id != pid]
        self.turn += 1
        # Reset per-turn dev restrictions
        player.devs_bought_this_turn = Counter()
        player.dev_played_this_turn = False
        self.log(f"\\n-- Turn {self.turn}: {player.name}'s turn (VP={player.vp}+{player.hidden_vp}*) --")
        self.snapshot(f"Start {player.name}")
        self.log_full_state("start_turn")
        roll = self.rng.randint(1,6) + self.rng.randint(1,6)
        self.log(f"Dice roll: {roll}")
        if roll == 7:
            self.handle_robber(player, others)
        else:
            self.distribute_resources(roll); self.snapshot(f"Production {roll}")
        bot.turn_actions(self, player, others)
        self.update_special_cards()
        # Log end-of-turn state prior to win check
        self.log_full_state("end_turn")
        if player.vp + player.hidden_vp >= self.target_vp:
            self.log(f"*** {player.name} wins! VP={player.vp} (+{player.hidden_vp} hidden) on turn {self.turn}. ***")
            self.snapshot(f"{player.name} wins")
            return True
        return False

    def play(self):
        self.setup_initial_placements()
        return self.play_loop()

    def play_loop(self):
        done = False
        while self.turn < self.max_turns and not done:
            if self.take_turn(self.current):
                done = True; break
            self.current = (self.current + 1) % len(self.players)
        if not done:
            scores = [(p.vp + p.hidden_vp, sum(p.hand.values()), -p.id) for p in self.players]
            winner_idx = max(range(len(self.players)), key=lambda i: scores[i])
            p = self.players[winner_idx]
            self.log(f"Max turns reached. {p.name} wins on tiebreaker (VP {p.vp}+{p.hidden_vp}, cards {sum(p.hand.values())}).")
            self.snapshot(f"Tiebreaker winner {p.name}")
        # Always record final full state
        self.log_full_state("final")
        return

class HeuristicBot:
    def __init__(self, player_id: int, name="Bot"):
        self.player_id = player_id; self.name = name

    def choose_initial_settlement(self, game, player, others):
        legal = game.legal_settlement_spots(player, others, require_connection=False)
        if not legal: return game.rng.choice(list(game.board.nodes.keys()))
        return max(legal, key=lambda n: game.node_expectation(n))

    def choose_initial_road(self, game, player, others, from_node):
        candidates = []
        # Disallow edges that touch an opponent building at either endpoint
        opponent_occupied = set()
        for p in game.players:
            if p.id == player.id:
                continue
            opponent_occupied |= set(p.settlements)
            opponent_occupied |= set(p.cities)
        for nb in node_neighbors(from_node, game.board.edges):
            e = add_edge((from_node, nb))
            u,v = e
            if u in opponent_occupied or v in opponent_occupied:
                continue
            candidates.append(e)
        if not candidates:
            return game.rng.choice(list(game.board.edges))
        # Prefer roads that point to an endpoint that can lead to a legal settlement spot
        def node_open_for_settlement(nid: int) -> bool:
            for p in game.players:
                if nid in p.settlements or nid in p.cities:
                    return False
            for m in game.board.adj.get(nid, ()):  # distance rule
                for p in game.players:
                    if m in p.settlements or m in p.cities:
                        return False
            return True
        def node_threat_level(nid: int) -> int:
            # How many opponents can reach nid in <=1 road step
            threat = 0
            for opp in game.players:
                if opp.id == player.id:
                    continue
                conn = game.player_connected_nodes(opp)
                # already has road touching nid?
                if any((u == nid or v == nid) for (u,v) in opp.roads):
                    threat += 2; continue
                for nb in game.board.adj.get(nid, ()):  # one road away
                    if nb in conn:
                        threat += 1; break
            return threat
        def edge_score(e):
            u,v = e; far = v if u == from_node else u
            # Base on far node expectation
            base = game.node_expectation(far)
            # Big boost if far node itself is open for settlement
            if node_open_for_settlement(far):
                base += 3.0
            else:
                # Look one step beyond for an open node
                nb_best = max([game.node_expectation(x) for x in node_neighbors(far, game.board.edges) if node_open_for_settlement(x)], default=0)
                base += 0.5 * nb_best
            # Penalize contested targets so multiple bots diversify
            threat = node_threat_level(far)
            base -= 1.2 * threat
            return base
        return max(candidates, key=edge_score)

    def best_trade_to_target(self, game, player, target_cost):
        # Try a small sequence of port/bank trades to finish the target build.
        need = Counter({r:max(0, c - player.hand[r]) for r,c in target_cost.items()})
        if sum(need.values()) == 0:
            return False
        # Attempt up to 2 trades
        did = False
        for _ in range(2):
            wants = [r for r,c in need.items() for _ in range(c)]
            if not wants: break
            want = wants[0]
            best_give = None; best_score = -1; best_rate = None
            for r,c in list(player.hand.items()):
                if r == want: continue
                rate = game.best_trade_rate(player, r)
                if c >= rate and rate <= 4:
                    score = (10 - rate) * 10 + c
                    if score > best_score:
                        best_score = score; best_give = r; best_rate = rate
            if best_give is None:
                break
            player.hand[best_give] -= best_rate
            if player.hand[best_give] == 0: del player.hand[best_give]
            player.hand[want] += 1
            game.log(f"{player.name} trades {best_rate}:1 via port/bank - gives {best_rate} {best_give} for 1 {want}.")
            game.snapshot(f"{player.name} trades {best_give}->{want}")
            need = Counter({r:max(0, target_cost[r] - player.hand[r]) for r in target_cost})
            did = True
            if sum(need.values()) == 0:
                break
        return did

    def maybe_buy_dev_and_play(self, game, player, others):
        want_set = game.legal_settlement_spots(player, others, require_connection=True)
        can_city = can_afford(player.hand, BUILD_COST["city"]) and len(player.settlements)>0 and player.cities_left>0
        can_settle = bool(want_set) and can_afford(player.hand, BUILD_COST["settlement"]) and player.settlements_left>0
        if not can_city and not can_settle:
            if can_afford(player.hand, BUILD_COST["dev"]): game.buy_dev_card(player)
        if player.dev_hand["knight"]>0:
            should = False
            for hid, h in game.board.hexes.items():
                if h.number in (6,8):
                    if any(any(n in opp.settlements or n in opp.cities for n in h.nodes) for opp in others):
                        should = True; break
            if should: game.play_knight(player, others)
        if player.dev_hand["year_of_plenty"]>0:
            need_city = Counter({r:max(0, BUILD_COST["city"][r]-player.hand[r]) for r in BUILD_COST["city"]})
            need_set = Counter({r:max(0, BUILD_COST["settlement"][r]-player.hand[r]) for r in BUILD_COST["settlement"]})
            if 0 < sum(need_city.values()) <= 2:
                picks = [r for r,c in need_city.items() for _ in range(c)]
                if len(picks)==1: picks.append(picks[0])
                game.play_year_of_plenty(player, picks[0], picks[1])
            elif 0 < sum(need_set.values()) <= 2:
                picks = [r for r,c in need_set.items() for _ in range(c)]
                if len(picks)==1: picks.append(picks[0])
                game.play_year_of_plenty(player, picks[0], picks[1])
        if player.dev_hand["monopoly"]>0:
            total_cards = sum(sum(opp.hand.values()) for opp in others)
            if total_cards>=4:
                res = max(RESOURCES, key=lambda r: sum(opp.hand[r] for opp in others))
                if sum(opp.hand[res] for opp in others) >= 3:
                    game.play_monopoly(player, others, res)
        if player.dev_hand["road_building"]>0 and player.roads_left>0:
            legal = game.legal_road_spots(player)
            if legal:
                game.play_road_building(player)

    def turn_actions(self, game, player, others):
        builds = 0
        # Try a P2P trade before building
        for opp in others:
            if game.p2p_trade(player, opp):
                break
        # Strategic road-building early: if we cannot afford a city right now,
        # build one road toward high-value nodes to extend reach.
        can_city_now = (len(player.settlements)>0 and player.cities_left>0 and can_afford(player.hand, BUILD_COST["city"]))
        if not can_city_now and player.roads_left>0 and can_afford(player.hand, BUILD_COST["road"]):
            legal_r = game.legal_road_spots(player)
            if legal_r:
                def node_open_for_settlement(nid: int) -> bool:
                    for p in game.players:
                        if nid in p.settlements or nid in p.cities:
                            return False
                    for m in game.board.adj.get(nid, ()):  # distance rule
                        for p in game.players:
                            if m in p.settlements or m in p.cities:
                                return False
                    return True
                def node_threat_level(nid: int) -> int:
                    threat = 0
                    for opp in game.players:
                        if opp.id == player.id:
                            continue
                        conn = game.player_connected_nodes(opp)
                        if any((u == nid or v == nid) for (u,v) in opp.roads):
                            threat += 2; continue
                        for nb in game.board.adj.get(nid, ()):  # one road away
                            if nb in conn:
                                threat += 1; break
                    return threat
                def road_score(e):
                    u,v = e
                    u_open = node_open_for_settlement(u); v_open = node_open_for_settlement(v)
                    if not (u_open or v_open):
                        # Look one step further to see if this road progresses toward any open node
                        def next_best(n):
                            return max([game.node_expectation(x) for x in game.board.adj.get(n, ()) if node_open_for_settlement(x)], default=0)
                        nb = max(next_best(u), next_best(v))
                        # Contested penalty from both ends (min threat among ends used)
                        thr = min(node_threat_level(u), node_threat_level(v))
                        return -1.0 + 0.5 * nb - 1.0 * thr
                    # Prefer edges whose endpoint is immediately open for a settlement
                    su = game.node_expectation(u) + (2.0 if u_open else 0.0) - 1.2 * node_threat_level(u)
                    sv = game.node_expectation(v) + (2.0 if v_open else 0.0) - 1.2 * node_threat_level(v)
                    return max(su, sv)
                best_r = max(legal_r, key=road_score)
                pay_cost(player.hand, BUILD_COST["road"]); player.roads.add(best_r); player.roads_left -= 1
                builds += 1
                game.log(f"{player.name} builds ROAD early toward expansion on edge {best_r}.")
                game.snapshot(f"{player.name} -> ROAD {best_r}")
        while True:
            target_node = max(list(player.settlements), key=lambda n: game.node_expectation(n), default=None)
            if target_node and can_afford(player.hand, BUILD_COST["city"]) and player.cities_left>0:
                pay_cost(player.hand, BUILD_COST["city"])
                player.settlements.remove(target_node); player.cities.add(target_node); player.vp += 1
                player.cities_left -= 1; player.settlements_left += 1
                builds += 1
                game.log(f"{player.name} upgrades to CITY at node {target_node}. VP={player.vp}")
                game.snapshot(f"{player.name} -> CITY {target_node}")
            else:
                deficit = sum(max(0, BUILD_COST["city"][r] - player.hand[r]) for r in BUILD_COST["city"])
                if target_node and 0 < deficit <= 2:
                    if not self.best_trade_to_target(game, player, BUILD_COST["city"]): break
                    else: continue
                break

        self.maybe_buy_dev_and_play(game, player, others)
        # Try a P2P trade again after devs
        for opp in others:
            if game.p2p_trade(player, opp):
                break

        while True:
            spots = game.legal_settlement_spots(player, others, require_connection=True)
            if spots and can_afford(player.hand, BUILD_COST["settlement"]) and player.settlements_left>0:
                nid = max(spots, key=lambda n: game.node_expectation(n))
                pay_cost(player.hand, BUILD_COST["settlement"])
                player.settlements.add(nid); player.vp += 1; player.settlements_left -= 1
                builds += 1
                game.log(f"{player.name} builds SETTLEMENT at node {nid}. VP={player.vp}")
                game.snapshot(f"{player.name} -> SETTLEMENT {nid}")
            else:
                if spots:
                    deficit = sum(max(0, BUILD_COST["settlement"][r] - player.hand[r]) for r in BUILD_COST["settlement"])
                    if 0 < deficit <= 2:
                        if self.best_trade_to_target(game, player, BUILD_COST["settlement"]): continue
                break

        target_nodes = game.legal_settlement_spots(player, others, require_connection=False)
        target = max(target_nodes, key=lambda n: game.node_expectation(n)) if target_nodes else None
        road_builds = 0
        while can_afford(player.hand, BUILD_COST["road"]) and player.roads_left>0:
            legal = game.legal_road_spots(player)
            if not legal: break
            def node_open_for_settlement(nid: int) -> bool:
                for p in game.players:
                    if nid in p.settlements or nid in p.cities:
                        return False
                for m in game.board.adj.get(nid, ()):  # distance rule
                    for p in game.players:
                        if m in p.settlements or m in p.cities:
                            return False
                return True
            def node_threat_level(nid: int) -> int:
                threat = 0
                for opp in game.players:
                    if opp.id == player.id:
                        continue
                    conn = game.player_connected_nodes(opp)
                    if any((u == nid or v == nid) for (u,v) in opp.roads):
                        threat += 2; continue
                    for nb in game.board.adj.get(nid, ()):  # one road away
                        if nb in conn:
                            threat += 1; break
                return threat
            def road_score(e):
                u,v = e
                u_open = node_open_for_settlement(u); v_open = node_open_for_settlement(v)
                if not (u_open or v_open):
                    def next_best(n):
                        return max([game.node_expectation(x) for x in game.board.adj.get(n, ()) if node_open_for_settlement(x)], default=0)
                    nb = max(next_best(u), next_best(v))
                    thr = min(node_threat_level(u), node_threat_level(v))
                    return -1.0 + 0.5 * nb - 1.0 * thr
                su = game.node_expectation(u) + (2.0 if u_open else 0.0) - 1.2 * node_threat_level(u)
                sv = game.node_expectation(v) + (2.0 if v_open else 0.0) - 1.2 * node_threat_level(v)
                return max(su, sv)
            choice = max(legal, key=road_score)
            pay_cost(player.hand, BUILD_COST["road"]); player.roads.add(choice); player.roads_left -= 1
            road_builds += 1; builds += 1
            game.log(f"{player.name} builds ROAD on edge {choice}.")
            game.snapshot(f"{player.name} -> ROAD {choice}")
            if road_builds >= 2: break

        if can_afford(player.hand, BUILD_COST["dev"]):
            # Reduce aggressive dev buying: only buy if it mitigates 7-discard risk
            # or if a tactical card (Road Building) is still in the deck and we can use roads.
            hand_total = sum(player.hand.values())
            risk_discard = hand_total > 7
            want_tactical = ("road_building" in game.dev_deck and player.roads_left > 0)
            if risk_discard or want_tactical:
                game.buy_dev_card(player)
        return builds

    def choose_robber_target_hex(self, game, player, others):
        # Robber targeting: avoid hurting yourself unless no alternative.
        leader = max(others, key=lambda p: (p.vp + p.hidden_vp, sum(p.hand.values())), default=None)
        candidates = []  # (hid, base_score, me_touch)
        for hid, h in game.board.hexes.items():
            if hid == game.board.robber_hex: continue
            if h.number is None: continue
            weight = DICE_WEIGHTS.get(h.number, 0)
            if weight == 0: continue
            opp_touch = 0; me_touch = 0; leader_touch = 0; victims = []
            for nid in h.nodes:
                for opp in others:
                    if nid in opp.settlements: opp_touch += 1; victims.append(opp)
                    if nid in opp.cities: opp_touch += 2; victims.append(opp)
                if nid in player.settlements: me_touch += 1
                if nid in player.cities: me_touch += 2
                if leader and (nid in leader.settlements or nid in leader.cities):
                    leader_touch += 1
            if opp_touch == 0:
                continue
            victim_hand = max((sum(v.hand.values()) for v in victims), default=0)
            base = 1.0 * weight * opp_touch + 0.6 * leader_touch + 0.05 * victim_hand
            candidates.append((hid, base, me_touch))
        if not candidates:
            return game.board.robber_hex
        # Prefer tiles that do not hurt us at all
        zero = [(hid, score) for (hid, score, me) in candidates if me == 0]
        if zero:
            return max(zero, key=lambda t: t[1])[0]
        # Otherwise, least self-harming with heavy penalty for self-touch
        penalized = [(hid, score - 2.5 * me) for (hid, score, me) in candidates]
        return max(penalized, key=lambda t: t[1])[0]

class MCTSBot(HeuristicBot):
    def turn_actions(self, game, player, others):
        # Beam search over short action sequences within the turn using a lightweight sim state.
        beam_width = getattr(game, 'mcts_beam_width', 6)
        max_depth = getattr(game, 'mcts_max_depth', 3)

        def clone_player(p: Player) -> Player:
            cp = Player(p.id, p.name)
            cp.vp = p.vp; cp.hidden_vp = p.hidden_vp
            cp.settlements = set(p.settlements); cp.cities = set(p.cities)
            cp.roads = set(p.roads); cp.hand = Counter(p.hand); cp.dev_hand = Counter(p.dev_hand)
            cp.knights_played = p.knights_played; cp.has_largest_army = p.has_largest_army
            cp.longest_road_len = p.longest_road_len; cp.has_longest_road = p.has_longest_road
            cp.roads_left = p.roads_left; cp.settlements_left = p.settlements_left; cp.cities_left = p.cities_left
            return cp

        def clone_others(ops: List[Player]) -> List[Player]:
            res = []
            for op in ops:
                co = Player(op.id, op.name)
                # For simulation, we need at least hands to evaluate Monopoly; keep board pieces unchanged
                co.hand = Counter(op.hand)
                co.settlements = set(op.settlements); co.cities = set(op.cities)
                co.roads = set(op.roads)
                co.vp = op.vp; co.hidden_vp = op.hidden_vp
                res.append(co)
            return res

        def make_sim_state():
            # Clone and shuffle a private dev deck to preserve hidden information
            sim_deck = list(game.dev_deck)
            try:
                import random as _rand
                _rand.shuffle(sim_deck)
            except Exception:
                pass
            return {
                'p': clone_player(player),
                'others': clone_others(others),
                'dev_deck': sim_deck,
            }

        def legal_actions(sim):
            p = sim['p']
            acts = []
            # build city
            if can_afford(p.hand, BUILD_COST["city"]) and p.settlements and p.cities_left>0:
                acts.append(("build_city", max(list(p.settlements), key=lambda n: game.node_expectation(n))))
            # build settlement
            spots = game.legal_settlement_spots(p, others, require_connection=True)
            if spots and can_afford(p.hand, BUILD_COST["settlement"]) and p.settlements_left>0:
                best_n = max(spots, key=lambda n: game.node_expectation(n)); acts.append(("build_settlement", best_n))
            # build road
            roads = game.legal_road_spots(p)
            if roads and can_afford(p.hand, BUILD_COST["road"]) and p.roads_left>0:
                def node_open_for_settlement(nid: int) -> bool:
                    for pl in game.players:
                        if nid in pl.settlements or nid in pl.cities:
                            return False
                    for m in game.board.adj.get(nid, ()):  # distance rule
                        for pl in game.players:
                            if m in pl.settlements or m in pl.cities:
                                return False
                    return True
                def score_edge(e):
                    u,v = e
                    u_open = node_open_for_settlement(u); v_open = node_open_for_settlement(v)
                    if not (u_open or v_open):
                        def next_best(n):
                            return max([game.node_expectation(x) for x in game.board.adj.get(n, ()) if node_open_for_settlement(x)], default=0)
                        nb = max(next_best(u), next_best(v))
                        return -1.0 + 0.5 * nb
                    return max(game.node_expectation(u) + (2.0 if u_open else 0.0),
                               game.node_expectation(v) + (2.0 if v_open else 0.0))
                acts.append(("build_road", max(roads, key=score_edge)))
            # buy dev
            if can_afford(p.hand, BUILD_COST["dev"]) and sim['dev_deck']:
                acts.append(("buy_dev", None))
            # play devs (simulate if available and not already played this turn in real game)
            if p.dev_hand["knight"]>0 and not player.dev_played_this_turn: acts.append(("play_knight", None))
            if p.dev_hand["year_of_plenty"]>0 and not player.dev_played_this_turn: acts.append(("play_yop", None))
            if p.dev_hand["monopoly"]>0 and not player.dev_played_this_turn: acts.append(("play_monopoly", None))
            if p.dev_hand["road_building"]>0 and p.roads_left>0 and not player.dev_played_this_turn: acts.append(("play_road_building", None))
            # bank/port smart trade proxies
            acts.append(("smart_trade_city", None))
            acts.append(("smart_trade_set", None))
            return acts

        def apply(act, sim):
            p = sim['p']
            kind, arg = act
            if kind == "build_city" and p.settlements and can_afford(p.hand, BUILD_COST["city"]) and p.cities_left>0:
                pay_cost(p.hand, BUILD_COST["city"]); p.cities_left-=1
                target = arg
                if target in p.settlements:
                    p.settlements.remove(target); p.cities.add(target); p.vp += 1
            elif kind == "build_settlement" and arg is not None and can_afford(p.hand, BUILD_COST["settlement"]) and p.settlements_left>0:
                pay_cost(p.hand, BUILD_COST["settlement"]); p.settlements_left-=1
                p.settlements.add(arg); p.vp += 1
            elif kind == "build_road" and arg is not None and can_afford(p.hand, BUILD_COST["road"]) and p.roads_left>0:
                pay_cost(p.hand, BUILD_COST["road"]); p.roads_left-=1; p.roads.add(arg)
            elif kind == "buy_dev" and can_afford(p.hand, BUILD_COST["dev"]) and sim['dev_deck']:
                pay_cost(p.hand, BUILD_COST["dev"])
                card = sim['dev_deck'].pop()  # draw from simulated deck
                p.dev_hand[card] += 1
            elif kind == "play_knight" and p.dev_hand["knight"]>0:
                p.dev_hand["knight"] -= 1; p.knights_played += 1
            elif kind == "play_yop" and p.dev_hand["year_of_plenty"]>0:
                p.dev_hand["year_of_plenty"] -= 1
                # Choose two resources that most reduce deficits toward city/settlement
                need_city = Counter({r:max(0, BUILD_COST["city"][r]-p.hand[r]) for r in BUILD_COST["city"]})
                need_set = Counter({r:max(0, BUILD_COST["settlement"][r]-p.hand[r]) for r in BUILD_COST["settlement"]})
                combined = Counter(need_city) + Counter(need_set)
                picks = [r for r,_ in combined.most_common(2)]
                if len(picks) == 1: picks.append(picks[0])
                for r in picks: p.hand[r] += 1
            elif kind == "play_monopoly" and p.dev_hand["monopoly"]>0:
                p.dev_hand["monopoly"] -= 1
                # Pick resource maximizing total in simulated opponents' hands, then transfer
                res = max(RESOURCES, key=lambda r: sum(op.hand[r] for op in sim['others']))
                take = 0
                for op in sim['others']:
                    amt = op.hand[res]
                    if amt > 0:
                        op.hand[res] -= amt
                        if op.hand[res] == 0: del op.hand[res]
                        p.hand[res] += amt
                        take += amt
                # no log in sim
            elif kind == "play_road_building" and p.dev_hand["road_building"]>0 and p.roads_left>0:
                p.dev_hand["road_building"] -= 1
                # Add up to two roads if possible; choose edges touching high-expectation nodes
                for _ in range(2):
                    if p.roads_left <= 0:
                        break
                    legal = game.legal_road_spots(p)
                    if not legal:
                        break
                    choice = max(legal, key=lambda e: max(game.node_expectation(e[0]), game.node_expectation(e[1])))
                    p.roads.add(choice); p.roads_left -= 1
            elif kind == "smart_trade_city":
                need = Counter({r:max(0, BUILD_COST["city"][r]-p.hand[r]) for r in BUILD_COST["city"]})
                if sum(need.values())>0:
                    want = max(need, key=lambda r: need[r])
                    for r in RESOURCES:
                        if r!=want and p.hand[r] >= 3:
                            p.hand[r]-=3; p.hand[want]+=1; break
            elif kind == "smart_trade_set":
                need = Counter({r:max(0, BUILD_COST["settlement"][r]-p.hand[r]) for r in BUILD_COST["settlement"]})
                if sum(need.values())>0:
                    want = max(need, key=lambda r: need[r])
                    for r in RESOURCES:
                        if r!=want and p.hand[r] >= 3:
                            p.hand[r]-=3; p.hand[want]+=1; break

        def eval_state(sim) -> float:
            p: Player = sim['p']
            others_sim: List[Player] = sim['others']
            base = p.vp + p.hidden_vp
            # Production value via node expectations
            prod = sum(game.node_expectation(n) for n in p.settlements | p.cities)
            # Resource production by type (for port synergy)
            res_prod = Counter()
            owned_nodes = set(p.settlements) | set(p.cities)
            for nid in owned_nodes:
                mult = 2 if nid in p.cities else (1 if nid in p.settlements else 0)
                for hid in game.board.node_to_hexes.get(nid, []):
                    h = game.board.hexes[hid]
                    if h.resource and h.number is not None and hid != game.board.robber_hex:
                        res_prod[h.resource] += DICE_WEIGHTS.get(h.number, 0) * mult
            # Port value: generic ports are modest; resource ports scale with production of that resource
            port_value = 0.0
            for nid in owned_nodes:
                if nid in game.board.ports:
                    pinfo = game.board.ports[nid]
                    if pinfo.get('type') == 'generic':
                        port_value += 0.15
                    elif pinfo.get('type') == 'resource':
                        r = pinfo.get('resource')
                        port_value += 0.02 * res_prod.get(r, 0)
            # Hand risk (7-discard pressure)
            handrisk = max(0, sum(p.hand.values())-7)
            # Dev utility
            dev_util = (p.dev_hand["knight"] + p.dev_hand["road_building"] + p.dev_hand["year_of_plenty"] + p.dev_hand["monopoly"]) * 0.05
            # Longest road potential bonus (small, encourages building toward 5)
            try:
                lr_len = compute_longest_road_length(game.board, p, others_sim)
            except Exception:
                lr_len = p.longest_road_len
            longest_road_potential = 0.03 * lr_len
            # Opponent threat penalty: slight penalty for trailing the leader
            leader_vp = max((op.vp + op.hidden_vp) for op in others_sim) if others_sim else 0
            vp_diff = max(0, leader_vp - (p.vp + p.hidden_vp))
            opp_threat_penalty = 0.10 * vp_diff
            return (
                base
                + 0.06 * prod
                + 0.02 * len(p.roads)
                + dev_util
                + port_value
                + longest_road_potential
                - 0.03 * handrisk
                - opp_threat_penalty
            )

        # Beam search
        beam = [([], make_sim_state())]
        for _ in range(max_depth):
            candidates = []
            for seq, sim in beam:
                for a in legal_actions(sim):
                    nsim = {
                        'p': clone_player(sim['p']),
                        'others': clone_others(sim['others']),
                        'dev_deck': list(sim['dev_deck']),
                    }
                    apply(a, nsim)
                    score = eval_state(nsim)
                    candidates.append((seq+[a], nsim, score))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = [(seq, st) for (seq, st, _) in candidates[:beam_width]]
            if not beam:
                break
        best_seq = beam[0][0] if beam else []

        # execute best sequence using real game functions
        for (kind, arg) in best_seq:
            if kind == "build_city" and can_afford(player.hand, BUILD_COST["city"]) and player.settlements:
                target = max(list(player.settlements), key=lambda n: game.node_expectation(n))
                pay_cost(player.hand, BUILD_COST["city"]); player.settlements.remove(target); player.cities.add(target); player.vp += 1
                game.log(f"{player.name}[MCTS] builds CITY at {target}"); game.snapshot(f"{player.name} MCTS CITY {target}")
            elif kind == "build_settlement" and arg is not None and can_afford(player.hand, BUILD_COST["settlement"]) and player.settlements_left>0:
                pay_cost(player.hand, BUILD_COST["settlement"]); player.settlements.add(arg); player.vp += 1; player.settlements_left-=1
                game.log(f"{player.name}[MCTS] builds SETTLEMENT at {arg}"); game.snapshot(f"{player.name} MCTS SETTLEMENT {arg}")
            elif kind == "build_road" and arg is not None and can_afford(player.hand, BUILD_COST["road"]) and player.roads_left>0:
                pay_cost(player.hand, BUILD_COST["road"]); player.roads.add(arg); player.roads_left-=1
                game.log(f"{player.name}[MCTS] builds ROAD {arg}"); game.snapshot(f"{player.name} MCTS ROAD {arg}")
            elif kind == "buy_dev":
                game.buy_dev_card(player)
            elif kind == "play_knight":
                game.play_knight(player, others)
            elif kind == "play_yop":
                need = Counter({r:3-player.hand[r] for r in RESOURCES}); picks = sorted(RESOURCES, key=lambda r: need[r], reverse=True)[:2]
                game.play_year_of_plenty(player, picks[0], picks[1])
            elif kind == "play_monopoly":
                res = max(RESOURCES, key=lambda r: sum(opp.hand[r] for opp in others))
                game.play_monopoly(player, others, res)
            elif kind == "play_road_building":
                game.play_road_building(player)
            elif kind == "smart_trade_city":
                self.best_trade_to_target(game, player, BUILD_COST["city"])
            elif kind == "smart_trade_set":
                self.best_trade_to_target(game, player, BUILD_COST["settlement"]) 
        return super().turn_actions(game, player, others)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vp", type=int, default=10)
    parser.add_argument("--mcts", action="store_true")
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--trace", type=str, default="trace.json")
    parser.add_argument("--players", type=int, default=2, help="Number of players (2-4)")
    parser.add_argument("--mcts-all", action="store_true", help="Use MCTS bot for all players")
    parser.add_argument("--games", type=int, default=1, help="Run multiple self-play games and report win rates")
    args = parser.parse_args()

    if args.games <= 1:
        g = Game(seed=args.seed, target_vp=args.vp, max_turns=args.turns, use_mcts_bot=args.mcts, num_players=args.players, mcts_all=args.mcts_all)
        g.play()
        with open("catan_plus_log.txt","w",encoding="utf-8") as f:
            for line in g.logs: f.write(line+"\n")
        with open(args.trace,"w",encoding="utf-8") as f:
            json.dump(g.trace, f)
        print("Done. Trace saved to", args.trace)
    else:
        # Multi-game self-play aggregate
        wins = [0 for _ in range(max(2, min(4, args.players)))]
        turns_total = 0
        for k in range(args.games):
            g = Game(seed=args.seed + k, target_vp=args.vp, max_turns=args.turns, use_mcts_bot=args.mcts, num_players=args.players, mcts_all=args.mcts_all)
            g.play()
            # Determine winner by final VP + hidden VP
            winner_idx = max(range(len(g.players)), key=lambda i: (g.players[i].vp + g.players[i].hidden_vp, sum(g.players[i].hand.values()), -i))
            wins[winner_idx] += 1
            turns_total += g.turn
        avg_turns = turns_total / args.games if args.games else None
        print("Self-play summary:")
        for i, w in enumerate(wins):
            rate = w / args.games
            print(f"  Player {i} wins: {w} ({rate:.1%})")
        print("  Avg turns:", avg_turns)
        # Simple Elo-like diff for 2 players
        if len(wins) == 2 and wins[0] > 0 and wins[1] > 0:
            import math
            elo_diff = 400 * math.log10(wins[0] / wins[1])
            print(f"  Approx Elo diff P0-P1: {elo_diff:.1f}")

