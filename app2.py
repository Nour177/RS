"""
VRPTW Research Dashboard
Aligned with RS_final.py (original backend — SAResult has no instrumented fields by default).
The instrumented subclasses live here and patch the result after the run.

FIX: build_heuristic_solution now wraps ALL heuristics uniformly.
     NN / Solomon / Tour Géant were passing backend.Customer objects directly
     to modules that expect their own Client-like objects — only Clarke-Wright
     (which uses a dict-based API) happened to work.  The fix converts the
     backend Instance into a thin Client-compatible wrapper before calling
     every heuristic, matching what test_RS.py already does correctly.
"""

import io
import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    px = go = None
    PLOTLY_OK = False

import RS_final as backend
import nearestNeighbor as nn
import solomon_inser as solomon
import tourGeant as tour
from regret_algorithm import algo as regret_algo
from regret_algorithm import clarke_wright as cw
from cooling_strategies.adaptive import TemperatureSchedule
from cooling_strategies.logarithmique import LogarithmicSchedule
from cooling_strategies.par_paliers import StepSchedule

BASE_DIR    = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "Archive"

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VRPTW Research Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.5px;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #cccccc;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.metric-label { color: #333333; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #000000; font-size: 1.6rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.metric-ok    { color: #2ecc71; }
.metric-fail  { color: #e74c3c; }
.section-tag {
    display: inline-block;
    background: #e0e0e0;
    color: #333333;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
}
.warning-box {
    background: #fff3cd;
    border-left: 3px solid #f39c12;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    color: #856404;
    font-size: 0.85rem;
    margin: 8px 0;
}
.info-box {
    background: #d1ecf1;
    border-left: 3px solid #2980b9;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    color: #0c5460;
    font-size: 0.85rem;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
def ensure_session_state():
    defaults = {
        "vrptw_results": [],
        "selected_result": None,
        "sa_params_confirmed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_session_state()


# ─────────────────────────────────────────────────────────────
#  CACHED HELPERS
# ─────────────────────────────────────────────────────────────
@st.cache_data
def list_instance_files() -> List[str]:
    if not ARCHIVE_DIR.exists():
        return []
    return sorted([p.name for p in ARCHIVE_DIR.iterdir()
                   if p.is_file() and p.suffix == ".txt"])


@st.cache_data
def load_instance(filepath: str) -> backend.Instance:
    return backend.read_solomon(str(ARCHIVE_DIR / filepath))


def normalize_route(route) -> List[int]:
    if not route:
        return [0, 0]
    if isinstance(route[0], int):
        if route[0] == 0 and route[-1] == 0:
            return list(route)
        return [0] + list(route) + [0]
    if hasattr(route[0], "id"):
        return [0] + [c.id for c in route] + [0]
    return [0] + list(route) + [0]


# ─────────────────────────────────────────────────────────────
#  CLIENT SHIM
#  A minimal object that satisfies the attribute interface
#  expected by nearestNeighbor, solomon_inser, and tourGeant.
#  test_RS.py calls these modules with its own Client class;
#  we replicate that here so app.py works identically.
# ─────────────────────────────────────────────────────────────
class _Client:
    """Thin shim — mirrors the Client class used in test_RS.py."""
    __slots__ = ("id", "x", "y", "demand", "ready_time", "due_time", "service_time")

    def __init__(self, id, x, y, demand, ready_time, due_time, service_time):
        self.id           = id
        self.x            = x
        self.y            = y
        self.demand       = demand
        self.ready_time   = ready_time
        self.due_time     = due_time          # note: due_time (not due_date)
        self.service_time = service_time


def _inst_to_clients(inst: backend.Instance):
    """
    Convert a backend Instance into (_depot, [_customers], capacity)
    using _Client shims — the exact types the heuristic modules expect.
    """
    d = inst.depot
    depot = _Client(
        id=d.id, x=d.x, y=d.y,
        demand=d.demand,
        ready_time=d.ready_time,
        due_time=d.due_date,          # backend uses due_date; shim uses due_time
        service_time=d.service_time,
    )
    customers = [
        _Client(
            id=c.id, x=c.x, y=c.y,
            demand=c.demand,
            ready_time=c.ready_time,
            due_time=c.due_date,
            service_time=c.service_time,
        )
        for c in inst.customers
    ]
    return depot, customers, inst.vehicle_capacity


def _euclidean(a: _Client, b: _Client) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _build_dict_inputs(depot: _Client, customers: List[_Client], vehicle_capacity: float):
    """
    Build the dict-based inputs (distance matrix, demandes, fenetres_temps,
    temps_service, tous_les_clients) shared by the Regret-K and Clarke-Wright
    wrappers — unchanged from test_RS.py.
    """
    n = len(customers)
    all_nodes = [depot] + customers

    matrice_distances = [
        [_euclidean(all_nodes[i], all_nodes[j]) for j in range(n + 1)]
        for i in range(n + 1)
    ]

    demandes      = {0: 0.0}
    fenetres_temps = {0: (depot.ready_time, depot.due_time)}
    temps_service  = {0: depot.service_time}

    for idx, c in enumerate(customers):
        cid = idx + 1
        demandes[cid]       = c.demand
        fenetres_temps[cid] = (c.ready_time, c.due_time)
        temps_service[cid]  = c.service_time

    tous_les_clients = list(range(1, n + 1))
    return matrice_distances, demandes, fenetres_temps, temps_service, tous_les_clients


def _raw_ids_to_int_routes(routes_ids, customers: List[_Client]) -> List[List[int]]:
    """
    Convert [[0, 3, 1, 0], ...] (client-id lists) → [[0, c.id, c.id, 0], ...]
    using the actual Customer .id values (not the 1-based index used internally
    by the regret/CW algorithms).
    """
    routes = []
    for route_ids in routes_ids:
        int_route = [0]
        for cid in route_ids[1:-1]:           # strip depot sentinels
            int_route.append(customers[cid - 1].id)
        int_route.append(0)
        if len(int_route) > 2:               # skip empty routes
            routes.append(int_route)
    return routes


# ─────────────────────────────────────────────────────────────
#  HEURISTIC BUILDER  (the fixed version)
# ─────────────────────────────────────────────────────────────
def build_heuristic_solution(
    heuristic: str,
    inst: backend.Instance,
    regret_k: int = 3,
    tour_alpha: float = 0.5,
    tour_beta: float = 0.5,
) -> List[List[int]]:
    """
    Build an initial solution using the requested heuristic and return it
    as a list of integer-id routes: [[0, c1, c2, ..., 0], ...].

    All heuristics now go through _Client shims so every module receives
    the object types it was written for — matching test_RS.py exactly.
    """
    depot, customers, capacity = _inst_to_clients(inst)

    # ── Heuristics that return List[List[_Client]] ────────────
    if heuristic == "Nearest Neighbor":
        raw = nn.initial_solution_nearest_neighbor(customers, depot, capacity)
        return [normalize_route(r) for r in raw]

    if heuristic == "Solomon Insertion":
        raw = solomon.initial_solution_solomon_insertion(customers, depot, capacity)
        return [normalize_route(r) for r in raw]

    if heuristic == "Tour Géant":
        raw = tour.initial_solution_hybrid_split(
            customers, depot, capacity,
            alpha=tour_alpha, beta=tour_beta,
        )
        return [normalize_route(r) for r in raw]

    # ── Heuristics that use the dict-based API ────────────────
    (dm, demandes, fenetres, svc,
     tous_clients) = _build_dict_inputs(depot, customers, capacity)

    if heuristic == "Regret-K":
        raw = regret_algo.generer_solution_initiale_randomisee(
            tous_clients, 0, dm, capacity,
            demandes, fenetres, svc, K=regret_k,
        )
        return _raw_ids_to_int_routes(raw, customers)

    if heuristic == "Clarke-Wright":
        raw = cw.generer_solution_clarke_wright(
            tous_clients, 0, dm, capacity,
            demandes, fenetres, svc,
        )
        return _raw_ids_to_int_routes(raw, customers)

    raise ValueError(f"Unknown heuristic: {heuristic}")


# ─────────────────────────────────────────────────────────────
#  INSTRUMENTED SUBCLASSES
# ─────────────────────────────────────────────────────────────
class InstrumentedSchedule(backend.TemperatureSchedule):
    _instances: List["InstrumentedSchedule"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature_history: List[float] = []
        self.accepted_history:    List[bool]  = []
        self.improved_history:    List[bool]  = []
        self.delta_history:       List[float] = []
        InstrumentedSchedule._instances.append(self)

    def accept(self, delta: float) -> bool:
        result = super().accept(delta)
        self.accepted_history.append(result)
        self.delta_history.append(delta)
        return result

    def record(self, accepted: bool, improved: bool):
        super().record(accepted, improved)
        self.improved_history.append(improved)
        self.temperature_history.append(self.T)


class InstrumentedSelector(backend.AdaptiveOperatorSelector):
    _instances: List["InstrumentedSelector"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_history:   List[Dict] = []
        self.operator_history: List[Dict] = []
        InstrumentedSelector._instances.append(self)

    def update(self, name: str, reward: float):
        super().update(name, reward)
        entry = {
            "iteration": len(self.weight_history) + 1,
            "name": name,
            "weights": self.weights.copy(),
            "uses":    self.uses.copy(),
            "scores":  self.scores.copy(),
        }
        self.operator_history.append(entry)
        self.weight_history.append(self.weights.copy())


# Instrumented versions of other cooling strategies
class InstrumentedLogarithmicSchedule(LogarithmicSchedule):
    """Instrumented wrapper for LogarithmicSchedule."""
    _instances: List["InstrumentedLogarithmicSchedule"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature_history: List[float] = []
        self.accepted_history:    List[bool]  = []
        self.improved_history:    List[bool]  = []
        self.delta_history:       List[float] = []
        InstrumentedLogarithmicSchedule._instances.append(self)

    def accept(self, delta: float) -> bool:
        result = super().accept(delta)
        self.accepted_history.append(result)
        self.delta_history.append(delta)
        return result

    def record(self, accepted: bool, improved: bool):
        super().record(accepted, improved)
        self.improved_history.append(improved)
        self.temperature_history.append(self.T)


class InstrumentedStepSchedule(StepSchedule):
    """Instrumented wrapper for StepSchedule."""
    _instances: List["InstrumentedStepSchedule"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature_history: List[float] = []
        self.accepted_history:    List[bool]  = []
        self.improved_history:    List[bool]  = []
        self.delta_history:       List[float] = []
        InstrumentedStepSchedule._instances.append(self)

    def accept(self, delta: float) -> bool:
        result = super().accept(delta)
        self.accepted_history.append(result)
        self.delta_history.append(delta)
        return result

    def record(self, accepted: bool, improved: bool):
        super().record(accepted, improved)
        self.improved_history.append(improved)
        self.temperature_history.append(self.T)


def run_sa_instrumented(
    initial_solution: List[List[int]],
    inst: backend.Instance,
    config: backend.SAConfig,
    custom_schedule = None,
) -> backend.SAResult:
    orig_sched = backend.TemperatureSchedule
    orig_sel   = backend.AdaptiveOperatorSelector

    backend.TemperatureSchedule      = InstrumentedSchedule
    backend.AdaptiveOperatorSelector = InstrumentedSelector
    InstrumentedSchedule._instances.clear()
    InstrumentedLogarithmicSchedule._instances.clear()
    InstrumentedStepSchedule._instances.clear()
    InstrumentedSelector._instances.clear()

    try:
        result = backend.simulated_annealing(initial_solution, inst, config, custom_schedule)

        # Retrieve instrumentation data from whichever schedule was used
        sched = None
        if InstrumentedSchedule._instances:
            sched = InstrumentedSchedule._instances[-1]
        elif InstrumentedLogarithmicSchedule._instances:
            sched = InstrumentedLogarithmicSchedule._instances[-1]
        elif InstrumentedStepSchedule._instances:
            sched = InstrumentedStepSchedule._instances[-1]
        
        sel = InstrumentedSelector._instances[-1] if InstrumentedSelector._instances else None

        if sched:
            result.temperature_history     = sched.temperature_history
            result.accepted_history        = sched.accepted_history
            result.improved_history        = sched.improved_history
            result.delta_history           = sched.delta_history
        if sel:
            result.operator_weight_history = sel.weight_history
            result.operator_update_history = sel.operator_history
        return result
    finally:
        backend.TemperatureSchedule      = orig_sched
        backend.AdaptiveOperatorSelector = orig_sel
        InstrumentedSchedule._instances.clear()
        InstrumentedLogarithmicSchedule._instances.clear()
        InstrumentedStepSchedule._instances.clear()
        InstrumentedSelector._instances.clear()


# ─────────────────────────────────────────────────────────────
#  DIAGNOSTICS
# ─────────────────────────────────────────────────────────────
def capture_diagnostics(solution: List[List[int]], inst: backend.Instance) -> str:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        backend.diagnose_initial_solution(solution, inst)
    finally:
        sys.stdout = old
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────────────────────
def make_result_dataframe(results: List[Dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Instance":       r["instance"],
        "Heuristic":      r["heuristic"],
        "Run":            r["run_index"],
        "Initial cost":   round(r["initial_cost"], 2),
        "Final cost":     round(r["final_cost"], 2),
        "Improvement %":  round(r["improvement_pct"], 2),
        "Init routes":    r["initial_routes"],
        "Final routes":   r["final_routes"],
        "Init feasible":  r["initial_feasible"],
        "Feasible":       r["feasible"],
        "Time (s)":       round(r["elapsed"], 2),

        
        "Final operator": r.get("final_operator", None),

    } for r in results])

def download_data(results: List[Dict], fmt: str) -> Tuple[bytes, str, str]:
    df = make_result_dataframe(results)
    if fmt == "csv":
        return df.to_csv(index=False).encode(), "vrptw_results.csv", "text/csv"
    return df.to_json(orient="records", indent=2).encode(), "vrptw_results.json", "application/json"


def compute_stagnation(improved_history: List[bool]) -> Tuple[int, int, Optional[int]]:
    best = cur = 0
    last_imp = None
    for idx, imp in enumerate(improved_history, 1):
        if imp:
            best = max(best, cur)
            cur  = 0
            last_imp = idx
        else:
            cur += 1
    return max(best, cur), cur, last_imp


# ─────────────────────────────────────────────────────────────
#  PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────
LIGHT_BG  = "#FFFFFF"
GRID_COL = "#E0E0E0"
TEXT_COL = "#000000"

def _base_layout(title: str, xlab: str = "", ylab: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(family="IBM Plex Mono", size=14, color=TEXT_COL)),
        paper_bgcolor=LIGHT_BG, plot_bgcolor=LIGHT_BG,
        font=dict(family="IBM Plex Sans", color=TEXT_COL),
        xaxis=dict(title=xlab, gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(title=ylab, gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        margin=dict(l=50, r=20, t=50, b=40),
    )


def plot_routes(
    inst: backend.Instance,
    initial: Optional[List[List[int]]],
    final:   Optional[List[List[int]]],
    show_initial: bool,
    show_final:   bool,
) -> "go.Figure":
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()

    if show_initial and initial:
        for idx, route in enumerate(initial):
            coords = [(inst.node(n).x, inst.node(n).y) for n in route]
            xs, ys = zip(*coords)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(color=colors[idx % len(colors)], dash="dot", width=1.5),
                marker=dict(size=5, opacity=0.6),
                name=f"Init R{idx+1}", legendgroup="initial",
                legendgrouptitle_text="Initial" if idx == 0 else "",
            ))

    if show_final and final:
        for idx, route in enumerate(final):
            coords = [(inst.node(n).x, inst.node(n).y) for n in route]
            xs, ys = zip(*coords)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(color=colors[idx % len(colors)], width=2.5),
                marker=dict(size=7),
                name=f"Final R{idx+1}", legendgroup="final",
                legendgrouptitle_text="Final" if idx == 0 else "",
            ))

    fig.add_trace(go.Scatter(
        x=[inst.depot.x], y=[inst.depot.y],
        mode="markers+text",
        marker=dict(size=18, color="#f1c40f", symbol="star"),
        text=["Depot"], textposition="top center",
        textfont=dict(family="IBM Plex Mono", size=11),
        name="Depot", showlegend=True,
    ))

    fig.update_layout(
        **_base_layout("Route Map", "X", "Y"),
        height=620,
        legend=dict(bgcolor=LIGHT_BG, bordercolor=GRID_COL, borderwidth=1,
                    font=dict(size=10, family="IBM Plex Mono")),
    )
    return fig


def plot_series(title: str, y: List[float], xlab: str = "Iteration",
                color: str = "#4fa3e0") -> "go.Figure":
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(y) + 1)), y=y,
        mode="lines", line=dict(color=color, width=1.5),
    ))
    fig.update_layout(**_base_layout(title, xlab, title), height=300)
    return fig


def plot_operator_bar(op_stats: dict) -> "go.Figure":
    names   = list(op_stats.keys())
    weights = [op_stats[n]["weight"] for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=weights,
        marker=dict(
            color=weights,
            colorscale="Blues",
            line=dict(color=GRID_COL, width=1),
        ),
        text=[f"{w:.4f}" for w in weights],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10),
    ))
    fig.update_layout(**_base_layout("Final Operator Weights", "Operator", "Weight"), height=320)
    return fig


def plot_operator_evolution(update_history: List[Dict], op_names: List[str]) -> "go.Figure":
    df = pd.DataFrame([
        {"Iteration": e["iteration"], **{n: e["weights"].get(n, 0) for n in op_names}}
        for e in update_history
    ])
    step = max(1, len(df) // 2000)
    df   = df.iloc[::step].reset_index(drop=True)

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, op in enumerate(op_names):
        if op in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Iteration"], y=df[op], mode="lines",
                name=op, line=dict(color=palette[i % len(palette)], width=1.8),
            ))
    fig.update_layout(**_base_layout("Operator Weight Evolution", "Update #", "Weight"),
                      height=340,
                      legend=dict(bgcolor=LIGHT_BG, bordercolor=GRID_COL, borderwidth=1))
    return fig


def plot_route_bar(route_df: pd.DataFrame, col: str, title: str, color: str) -> "go.Figure":
    fig = go.Figure(go.Bar(
        x=route_df["Route"].astype(str), y=route_df[col],
        marker=dict(color=color, line=dict(color=TEXT_COL, width=1)),
        text=[f"{v:.1f}" for v in route_df[col]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9),
    ))
    fig.update_layout(**_base_layout(title, "Route", col), height=300)
    return fig


def plot_convergence(history: List[float]) -> "go.Figure":
    x = [(i + 1) * 1000 for i in range(len(history))]
    fig = go.Figure(go.Scatter(
        x=x, y=history, mode="lines+markers",
        line=dict(color="#2ecc71", width=2),
        marker=dict(size=4, color="#2ecc71"),
    ))
    fig.update_layout(**_base_layout("Best Cost Convergence", "Iteration", "Best Cost"),
                      height=320)
    return fig


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## VRPTW Dashboard")
    st.markdown("---")

    st.markdown("### Instances")
    instance_files = list_instance_files()
    if not instance_files:
        st.error("No `.txt` files found in `Archive/`.")
    selected_files = st.multiselect(
        "Instance files", instance_files,
        default=instance_files[:1] if instance_files else [],
    )

    st.markdown("### Heuristics")
    heuristics = st.multiselect(
        "Initial solution algorithms",
        ["Nearest Neighbor", "Solomon Insertion", "Tour Géant", "Regret-K", "Clarke-Wright"],
        default=["Nearest Neighbor"],
    )

    use_regret = "Regret-K" in heuristics
    use_tour   = "Tour Géant" in heuristics

    if use_regret:
        st.markdown("#### Regret-K parameters")
        regret_k = st.slider("K value", 1, 5, 3)
    else:
        regret_k = 3

    if use_tour:
        st.markdown("#### Tour Géant parameters")
        tour_alpha = st.slider("Alpha (time weight)", 0.0, 1.0, 0.5, step=0.05)
        tour_beta  = st.slider("Beta  (dist weight)", 0.0, 1.0, 0.5, step=0.05)
    else:
        tour_alpha = tour_beta = 0.5

    st.markdown("---")

    st.markdown("### Simulated Annealing")

    st.markdown("#### Cooling Strategy")
    cooling_strategy = st.radio(
        "Select cooling strategy",
        ["Adaptive (default)", "Logarithmic", "Step-based"],
        horizontal=False,
        help="Choose the temperature cooling schedule strategy.",
    )

    t_init_mode = st.radio(
        "T_init mode",
        ["Manual", "Auto-calibrate"],
        horizontal=True,
        help="Auto-calibrate estimates T_init from the initial solution.",
    )
    if t_init_mode == "Manual":
        t_init = st.number_input("T_init", min_value=0.1, value=100.0, step=10.0)
        acceptance_factor = 0.8  # Default, not used in manual mode
    else:
        t_init = None
        acceptance_factor = st.slider(
            "Acceptance factor (for auto-calibration)",
            min_value=0.1, max_value=0.99, value=0.8, step=0.05,
            help="Controls target acceptance rate for initial temperature estimation. Higher = warmer initial temp."
        )

    t_min          = st.number_input("T_min", min_value=1e-6, value=0.01, step=0.01, format="%.4f")
    max_iter       = st.number_input("Max iterations", min_value=1000, value=20000, step=1000)
    penalty_weight = st.number_input("Penalty weight", min_value=1.0, value=1000.0, step=100.0)

    with st.expander("Advanced SA parameters", expanded=False):
        if cooling_strategy == "Adaptive (default)":
            st.markdown("##### Adaptive Strategy Parameters")
            alpha                = st.slider("Cooling alpha (α)", min_value=0.90, max_value=0.9999,
                                            value=0.995, step=0.001, format="%.4f")
            target_acceptance    = st.slider("Target acceptance rate", 0.0, 1.0, 0.20, step=0.05)
            adapt_interval       = st.number_input("Adapt interval", min_value=50, value=500, step=50)
            reheat_patience      = st.number_input("Reheat patience", min_value=100, value=2000, step=100)
            reheat_factor        = st.number_input("Reheat factor", min_value=1.0, value=2.0, step=0.1)
            reaction_factor      = st.slider("Operator reaction factor", 0.0, 1.0, 0.15, step=0.05)
            segment_update       = st.number_input("Segment update", min_value=50, value=200, step=50)
            log_interval         = st.number_input("Log interval", min_value=100, value=10000, step=500)
            # Store strategy-specific params
            strategy_params = {
                "alpha": alpha,
                "target_acceptance": target_acceptance,
                "adapt_interval": adapt_interval,
                "reheat_patience": reheat_patience,
                "reheat_factor": reheat_factor,
            }
        elif cooling_strategy == "Logarithmic":
            st.markdown("##### Logarithmic Strategy Parameters")
            cooling_factor       = st.slider("Cooling factor (controls cooling speed)", 
                                            min_value=0.5, max_value=3.0, value=1.0, step=0.1, 
                                            format="%.1f",
                                            help="Higher values = faster cooling. T(k) = T_init / (factor * ln(1 + k))")
            alpha                = 0.995  # Not used, but keep for compatibility
            target_acceptance    = 0.20
            adapt_interval       = 500
            reheat_patience      = 2000
            reheat_factor        = 2.0
            reaction_factor      = st.slider("Operator reaction factor", 0.0, 1.0, 0.15, step=0.05)
            segment_update       = st.number_input("Segment update", min_value=50, value=200, step=50)
            log_interval         = st.number_input("Log interval", min_value=100, value=10000, step=500)
            strategy_params = {"cooling_factor": cooling_factor}
        else:  # Step-based
            st.markdown("##### Step-based Strategy Parameters")
            alpha                = st.slider("Cooling alpha (α) per step", min_value=0.50, max_value=0.99,
                                            value=0.90, step=0.01, format="%.2f")
            longueur_palier      = st.number_input("Step length (iterations per cooling)", 
                                                   min_value=100, value=1000, step=100)
            target_acceptance    = 0.20
            adapt_interval       = 500
            reheat_patience      = 2000
            reheat_factor        = 2.0
            reaction_factor      = st.slider("Operator reaction factor", 0.0, 1.0, 0.15, step=0.05)
            segment_update       = st.number_input("Segment update", min_value=50, value=200, step=50)
            log_interval         = st.number_input("Log interval", min_value=100, value=10000, step=500)
            strategy_params = {
                "alpha": alpha,
                "longueur_palier": longueur_palier,
            }

    st.markdown("---")

    st.markdown("### Run settings")
    runs_per_heuristic = st.number_input("Runs per heuristic", min_value=1, max_value=10, value=1)
    deterministic      = st.checkbox("Deterministic seed", value=False)
    random_seed        = st.number_input("Random seed", value=42, step=1)

    st.markdown("---")

    run_clicked   = st.button("  Run experiments", use_container_width=True, type="primary")
    clear_clicked = st.button("  Clear results",   use_container_width=True)

    if st.session_state.vrptw_results:
        st.markdown("---")
        st.markdown("### Export")
        export_fmt = st.radio("Format", ["csv", "json"], horizontal=True)
        data_bytes, fname, mime = download_data(st.session_state.vrptw_results, export_fmt)
        st.download_button("⬇  Download results", data=data_bytes,
                           file_name=fname, mime=mime, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  CLEAR
# ─────────────────────────────────────────────────────────────
if clear_clicked:
    st.session_state.vrptw_results   = []
    st.session_state.selected_result = None
    st.success("Results cleared.")


# ─────────────────────────────────────────────────────────────
#  RUN EXPERIMENTS
# ─────────────────────────────────────────────────────────────
if run_clicked:
    if not selected_files:
        st.warning("Select at least one instance file.")
    elif not heuristics:
        st.warning("Select at least one heuristic.")
    else:
        total_runs = len(selected_files) * len(heuristics) * int(runs_per_heuristic)
        progress   = st.progress(0.0, text="Starting…")
        run_count  = 0

        for instance_file in selected_files:
            inst = load_instance(instance_file)

            for heuristic in heuristics:
                for run_idx in range(1, int(runs_per_heuristic) + 1):

                    if deterministic:
                        random.seed(int(random_seed) + run_count)

                    label = f"{heuristic} | {instance_file} | run {run_idx}/{int(runs_per_heuristic)}"
                    progress.progress(run_count / total_runs, text=label)

                    with st.spinner(label):
                        try:
                            initial_sol = build_heuristic_solution(
                                heuristic, inst,
                                regret_k=regret_k,
                                tour_alpha=tour_alpha,
                                tour_beta=tour_beta,
                            )
                        except Exception as e:
                            st.error(f"Heuristic failed ({heuristic} / {instance_file}): {e}")
                            run_count += 1
                            continue

                        init_cost     = backend.total_cost(initial_sol, inst, penalty_weight)
                        init_feasible = backend.check_faisabilite(initial_sol, inst)

                        effective_t_init = (
                            backend.estimate_initial_temperature(
                                initial_sol, inst, target_accept=acceptance_factor)
                            if t_init is None
                            else t_init
                        )

                        config = backend.SAConfig(
                            T_init            = effective_t_init,
                            T_min             = float(t_min),
                            alpha             = float(alpha),
                            max_iter          = int(max_iter),
                            penalty_weight    = float(penalty_weight),
                            target_acceptance = float(target_acceptance),
                            adapt_interval    = int(adapt_interval),
                            reheat_patience   = int(reheat_patience),
                            reheat_factor     = float(reheat_factor),
                            reaction_factor   = float(reaction_factor),
                            segment_update    = int(segment_update),
                            verbose           = False,
                            log_interval      = int(log_interval),

                        )

                        # Create custom cooling strategy based on user selection
                        custom_schedule = None
                        strategy_info = {"name": cooling_strategy, "params": strategy_params.copy()}
                        
                        if cooling_strategy == "Adaptive (default)":
                            custom_schedule = InstrumentedSchedule(
                                T_init=effective_t_init,
                                T_min=float(t_min),
                                alpha=float(strategy_params.get("alpha", alpha)),
                                target_acceptance=float(strategy_params.get("target_acceptance", 0.2)),
                                adapt_interval=int(strategy_params.get("adapt_interval", 500)),
                                reheat_factor=float(strategy_params.get("reheat_factor", 2.0)),
                                reheat_patience=int(strategy_params.get("reheat_patience", 2000)),
                            )
                        elif cooling_strategy == "Logarithmic":
                            custom_schedule = InstrumentedLogarithmicSchedule(
                                T_init=effective_t_init,
                                T_min=float(t_min),
                                cooling_factor=float(strategy_params.get("cooling_factor", 1.0)),
                            )
                        elif cooling_strategy == "Step-based":
                            custom_schedule = InstrumentedStepSchedule(
                                T_init=effective_t_init,
                                T_min=float(t_min),
                                alpha=float(strategy_params.get("alpha", 0.90)),
                                longueur_palier=int(strategy_params.get("longueur_palier", 1000)),
                            )

                        sa_result = run_sa_instrumented(initial_sol, inst, config, custom_schedule)
                        # Extract dominant (final) operator
                        final_operator = None
                        if sa_result.operator_stats:
                            final_operator = max(
                                sa_result.operator_stats,
                                key=lambda k: sa_result.operator_stats[k]["weight"]
                            )

                    record = {
                        "instance":         instance_file,
                        "heuristic":        heuristic,
                        "run_index":        run_idx,
                        "initial_solution": initial_sol,
                        "final_solution":   sa_result.best_solution,
                        "initial_cost":     init_cost,
                        "final_cost":       sa_result.best_cost,
                        "improvement_pct":  0.0 if init_cost == 0 else
                            100.0 * (init_cost - sa_result.best_cost) / init_cost,
                        "initial_routes":   len(initial_sol),
                        "final_routes":     len(sa_result.best_solution),
                        "feasible":         sa_result.feasible,
                        "elapsed":          sa_result.elapsed,
                        "initial_feasible": init_feasible,
                        "sa_result":        sa_result,
                        "instance_obj":     inst,
                        "config":           config,
                        "t_init_used":      effective_t_init,
                        "cooling_strategy": strategy_info,

                        # ✅ NEW FIELD
                        "final_operator": final_operator,
                    }
                    st.session_state.vrptw_results.append(record)
                    run_count += 1

        progress.progress(1.0, text="Done ")
        st.success(f"Completed {total_runs} experiment(s).")


# ─────────────────────────────────────────────────────────────
#  MAIN TITLE
# ─────────────────────────────────────────────────────────────
st.markdown("# VRPTW Research Dashboard")
st.markdown(
    "<p style='color:#333333; font-family:IBM Plex Mono; font-size:0.85rem;'>"
    "Heuristic construction · Simulated Annealing · Constraint diagnostics"
    "</p>", unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs(["Experiments", "Routes", "Comparison", "SA Analysis", "Diagnostics"])


# ══════════════════════════════════════════════════════════════
#  TAB 0 — EXPERIMENTS
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-tag">experiment log</div>', unsafe_allow_html=True)
    st.markdown(f"**{len(st.session_state.vrptw_results)}** result(s) stored.")

    if not st.session_state.vrptw_results:
        st.markdown(
            '<div class="info-box">Configure parameters in the sidebar and click <b> Run experiments</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        df = make_result_dataframe(st.session_state.vrptw_results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        best_row = df.loc[df["Final cost"].idxmin()]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Best final cost</div>'
                f'<div class="metric-value">{best_row["Final cost"]:.1f}</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Best heuristic</div>'
                f'<div class="metric-value" style="font-size:1rem">{best_row["Heuristic"]}</div></div>',
                unsafe_allow_html=True)
        with c3:
            feas_count = df["Feasible"].sum()
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Feasible runs</div>'
                f'<div class="metric-value metric-ok">{feas_count} / {len(df)}</div></div>',
                unsafe_allow_html=True)
        with c4:
            avg_imp = df["Improvement %"].mean()
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Avg improvement</div>'
                f'<div class="metric-value">{avg_imp:.1f}%</div></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 1 — ROUTES
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-tag">route visualisation</div>', unsafe_allow_html=True)

    if not st.session_state.vrptw_results:
        st.markdown('<div class="info-box">Run experiments first.</div>', unsafe_allow_html=True)
    elif not PLOTLY_OK:
        st.error("Install plotly: `pip install plotly`")
    else:
        labels = [
            f"{r['instance']} | {r['heuristic']} | Run {r['run_index']}"
            for r in st.session_state.vrptw_results
        ]
        sel_label = st.selectbox("Select result", labels, key="route_selector")
        sel_idx   = labels.index(sel_label)
        st.session_state.selected_result = sel_idx

        result = st.session_state.vrptw_results[sel_idx]
        inst   = result["instance_obj"]

        col_l, col_r = st.columns([3, 1])
        with col_r:
            show_init  = st.checkbox("Show initial routes", value=True)
            show_final = st.checkbox("Show final routes",   value=True)
            st.markdown("---")
            imp_col  = "metric-ok" if result["improvement_pct"] >= 0 else "metric-fail"
            feas_col = "metric-ok" if result["feasible"] else "metric-fail"
            for label, val in [
                ("Initial cost",  f'{result["initial_cost"]:.1f}'),
                ("Final cost",    f'{result["final_cost"]:.1f}'),
            ]:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{val}</div></div>',
                    unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Improvement</div>'
                f'<div class="metric-value {imp_col}">{result["improvement_pct"]:.1f}%</div></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Feasible</div>'
                f'<div class="metric-value {feas_col}">{"Yes" if result["feasible"] else "No"}</div></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Routes</div>'
                f'<div class="metric-value">{result["initial_routes"]} → {result["final_routes"]}</div></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Time</div>'
                f'<div class="metric-value">{result["elapsed"]:.1f}s</div></div>',
                unsafe_allow_html=True)

        with col_l:
            fig = plot_routes(
                inst,
                result["initial_solution"] if show_init  else None,
                result["final_solution"]   if show_final else None,
                show_init, show_final,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2 — COMPARISON
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-tag">comparison dashboard</div>', unsafe_allow_html=True)

    if not st.session_state.vrptw_results:
        st.markdown('<div class="info-box">Run experiments first.</div>', unsafe_allow_html=True)
    else:
        df = make_result_dataframe(st.session_state.vrptw_results)

        if PLOTLY_OK:
            c1, c2 = st.columns(2)
            with c1:
                fig_imp = px.bar(
                    df, x="Heuristic", y="Improvement %", color="Instance",
                    barmode="group", title="Improvement % by heuristic",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                fig_imp.update_layout(**_base_layout("Improvement % by heuristic",
                                                     "Heuristic", "Improvement %"), height=360)
                st.plotly_chart(fig_imp, use_container_width=True)

            with c2:
                fig_cost = px.bar(
                    df, x="Heuristic", y="Final cost", color="Instance",
                    barmode="group", title="Final cost by heuristic",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                fig_cost.update_layout(**_base_layout("Final cost by heuristic",
                                                      "Heuristic", "Final cost"), height=360)
                st.plotly_chart(fig_cost, use_container_width=True)

            fig_sc = px.scatter(
                df, x="Initial cost", y="Final cost",
                color="Heuristic", symbol="Instance",
                hover_data=["Run", "Improvement %", "Feasible"],
                title="Initial vs Final cost",
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig_sc.update_layout(**_base_layout("Initial vs Final cost",
                                                "Initial cost", "Final cost"), height=380)
            mn = min(df["Initial cost"].min(), df["Final cost"].min())
            mx = max(df["Initial cost"].max(), df["Final cost"].max())
            fig_sc.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx], mode="lines",
                line=dict(color="#e74c3c", dash="dash", width=1),
                name="No improvement", showlegend=True,
            ))
            st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("#### Ranking (by final cost)")
        ranking = df.sort_values(["Final cost", "Improvement %"], ascending=[True, False])
        st.dataframe(ranking.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  TAB 3 — SA ANALYSIS
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-tag">simulated annealing analysis</div>', unsafe_allow_html=True)

    if st.session_state.selected_result is None or not st.session_state.vrptw_results:
        st.markdown(
            '<div class="info-box">Select a result in the <b> Routes</b> tab first.</div>',
            unsafe_allow_html=True,
        )
    else:
        result    = st.session_state.vrptw_results[st.session_state.selected_result]
        sa_result = result["sa_result"]
        inst      = result["instance_obj"]

        cfg = result.get("config")
        if cfg is not None:
            with st.expander("SA parameters used for this run", expanded=False):
                param_df = pd.DataFrame([{
                    "T_init":            result.get("t_init_used", cfg.T_init),
                    "T_min":             cfg.T_min,
                    "alpha":             cfg.alpha,
                    "max_iter":          cfg.max_iter,
                    "penalty_weight":    cfg.penalty_weight,
                    "target_acceptance": cfg.target_acceptance,
                    "adapt_interval":    cfg.adapt_interval,
                    "reheat_patience":   cfg.reheat_patience,
                    "reheat_factor":     cfg.reheat_factor,
                    "reaction_factor":   cfg.reaction_factor,
                    "segment_update":    cfg.segment_update,
                }])
                st.dataframe(param_df.T.rename(columns={0: "Value"}),
                             use_container_width=True)

            # Display cooling strategy information
            cooling_strat = result.get("cooling_strategy", {})
            if cooling_strat:
                with st.expander("Cooling strategy configuration", expanded=False):
                    st.markdown(f"**Strategy:** {cooling_strat.get('name', 'Unknown')}")
                    if cooling_strat.get("params"):
                        st.markdown("**Strategy-specific parameters:**")
                        strat_df = pd.DataFrame([cooling_strat["params"]]).T.rename(columns={0: "Value"})
                        st.dataframe(strat_df, use_container_width=True)
                    else:
                        st.markdown("*(No additional parameters for this strategy)*")

        st.markdown("---")

        st.markdown("#### Best-cost convergence")
        if sa_result.history:
            st.plotly_chart(plot_convergence(sa_result.history), use_container_width=True)
        else:
            st.markdown('<div class="warning-box">No convergence history available.</div>',
                        unsafe_allow_html=True)

        temp_hist = getattr(sa_result, "temperature_history", [])
        acc_hist  = getattr(sa_result, "accepted_history",    [])
        imp_hist  = getattr(sa_result, "improved_history",    [])

        if temp_hist:
            st.markdown("#### Temperature schedule")
            step = max(1, len(temp_hist) // 5000)
            st.plotly_chart(
                plot_series("Temperature over iterations", temp_hist[::step],
                            xlab="Iteration (sampled)", color="#e67e22"),
                use_container_width=True,
            )

            if acc_hist:
                st.markdown("#### Acceptance rate (rolling 200)")
                acc_series = (
                    pd.Series(acc_hist[::step])
                    .rolling(window=200, min_periods=1)
                    .mean()
                    .tolist()
                )
                st.plotly_chart(
                    plot_series("Acceptance rate", acc_series,
                                xlab="Iteration (sampled)", color="#9b59b6"),
                    use_container_width=True,
                )

            if imp_hist:
                st.markdown("#### Stagnation analysis")
                best_streak, cur_streak, last_imp = compute_stagnation(imp_hist)
                s1, s2, s3 = st.columns(3)
                s1.metric("Longest no-improvement streak", f"{best_streak:,}")
                s2.metric("Last improvement at iter",      f"{last_imp or 'N/A'}")
                s3.metric("Current streak at end",         f"{cur_streak:,}")
        else:
            st.markdown(
                '<div class="warning-box">Temperature/acceptance history not available '
                '(instrumentation issue).</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown("#### Final operator weights")
        if sa_result.operator_stats:
            if PLOTLY_OK:
                st.plotly_chart(plot_operator_bar(sa_result.operator_stats),
                                use_container_width=True)
            op_df = pd.DataFrame([
                {"Operator": k, "Final weight": v["weight"]}
                for k, v in sa_result.operator_stats.items()
            ]).sort_values("Final weight", ascending=False)
            st.dataframe(op_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No operator stats available.")

        op_upd = getattr(sa_result, "operator_update_history", [])
        if op_upd and PLOTLY_OK:
            st.markdown("#### Operator weight evolution")
            op_names = list(sa_result.operator_stats.keys())
            st.plotly_chart(plot_operator_evolution(op_upd, op_names),
                            use_container_width=True)

        st.markdown("---")

        st.markdown("#### Per-route analysis (final solution)")
        final_sol = result["final_solution"]
        route_df  = pd.DataFrame({
            "Route":     list(range(1, len(final_sol) + 1)),
            "Load":      [sum(inst.node(i).demand for i in r if i != 0) for r in final_sol],
            "Customers": [len(r) - 2 for r in final_sol],
            "Distance":  [backend.evaluate_route(r, inst)[0] for r in final_sol],
            "Penalty":   [backend.evaluate_route(r, inst)[1] for r in final_sol],
        })
        st.dataframe(route_df, use_container_width=True, hide_index=True)

        if PLOTLY_OK:
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.plotly_chart(
                    plot_route_bar(route_df, "Load",      "Load per vehicle",    "#3498db"),
                    use_container_width=True)
            with rc2:
                st.plotly_chart(
                    plot_route_bar(route_df, "Customers", "Customers per route", "#2ecc71"),
                    use_container_width=True)
            with rc3:
                st.plotly_chart(
                    plot_route_bar(route_df, "Distance",  "Distance per route",  "#e67e22"),
                    use_container_width=True)

        cap = inst.vehicle_capacity
        route_df["Util %"] = (route_df["Load"] / cap * 100).round(1)
        overloaded = route_df[route_df["Load"] > cap]
        if not overloaded.empty:
            st.markdown(
                f'<div class="warning-box">⚠ {len(overloaded)} route(s) exceed vehicle capacity '
                f'({cap:.0f}): routes {list(overloaded["Route"].values)}</div>',
                unsafe_allow_html=True)
        else:
            avg_util = route_df["Util %"].mean()
            st.markdown(
                f'<div class="info-box"> All routes within capacity. '
                f'Avg utilisation: {avg_util:.1f}%</div>',
                unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("#### Interpretation")
        dominant_op = max(sa_result.operator_stats, key=lambda k: sa_result.operator_stats[k]["weight"])
        dominant_w  = sa_result.operator_stats[dominant_op]["weight"]
        feasibility_str = " **feasible**" if result["feasible"] else " **infeasible**"
        stag_comment = (
            f"The longest stagnation streak was **{best_streak:,}** iterations."
            if imp_hist else "Stagnation data unavailable."
        )

        st.markdown(
            f"- SA improved cost by **{result['improvement_pct']:.2f}%** "
            f"({result['initial_cost']:.1f} → {result['final_cost']:.1f}).\n"
            f"- Final solution is {feasibility_str}.\n"
            f"- Dominant operator: **{dominant_op}** (weight `{dominant_w:.4f}`).\n"
            f"- {stag_comment}\n"
            f"- T_init used: **{result.get('t_init_used', 'N/A'):.2f}**, "
            f"final routes: **{result['final_routes']}**."
        )


# ══════════════════════════════════════════════════════════════
#  TAB 4 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-tag">constraint diagnostics</div>', unsafe_allow_html=True)

    if st.session_state.selected_result is None or not st.session_state.vrptw_results:
        st.markdown(
            '<div class="info-box">Select a result in the <b> Routes</b> tab first.</div>',
            unsafe_allow_html=True,
        )
    else:
        result = st.session_state.vrptw_results[st.session_state.selected_result]
        inst   = result["instance_obj"]

        d_col, f_col = st.columns(2)

        with d_col:
            st.markdown("**Initial solution**")
            init_feas = result["initial_feasible"]
            badge_col = "#2ecc71" if init_feas else "#e74c3c"
            st.markdown(
                f'<span style="background:{badge_col};color:#fff;padding:2px 8px;'
                f'border-radius:4px;font-family:IBM Plex Mono;font-size:0.75rem;">'
                f'{"FEASIBLE" if init_feas else "INFEASIBLE"}</span>',
                unsafe_allow_html=True,
            )
            diag_init = capture_diagnostics(result["initial_solution"], inst)
            st.code(diag_init, language=None)

        with f_col:
            st.markdown("**Final solution**")
            final_feas = result["feasible"]
            badge_col  = "#2ecc71" if final_feas else "#e74c3c"
            st.markdown(
                f'<span style="background:{badge_col};color:#fff;padding:2px 8px;'
                f'border-radius:4px;font-family:IBM Plex Mono;font-size:0.75rem;">'
                f'{"FEASIBLE" if final_feas else "INFEASIBLE"}</span>',
                unsafe_allow_html=True,
            )
            diag_final = capture_diagnostics(result["final_solution"], inst)
            st.code(diag_final, language=None)

        st.markdown("#### Constraint summary — final solution")
        rows = []
        for r_idx, route in enumerate(result["final_solution"]):
            load = sum(inst.node(i).demand for i in route if i != 0)
            _, pen, feas = backend.evaluate_route(route, inst)
            rows.append({
                "Route":          r_idx + 1,
                "Customers":      len(route) - 2,
                "Load":           round(load, 1),
                "Capacity OK":    load <= inst.vehicle_capacity,
                "Penalty":        round(pen, 2),
                "Route feasible": feas,
            })
        cstr_df = pd.DataFrame(rows)
        st.dataframe(cstr_df, use_container_width=True, hide_index=True)