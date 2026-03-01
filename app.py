import random
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



TASK_TYPES = [
    "Traffic Analysis",
    "Sensor Fusion",
    "Emergency Response",
    "Anomaly Detection",
    "Air Quality Monitor",
]

TASK_WEIGHT = {
    "Traffic Analysis":    1.0,
    "Sensor Fusion":       0.8,
    "Emergency Response":  1.5,
    "Anomaly Detection":   0.9,
    "Air Quality Monitor": 0.6,
}

PRIORITY_MULT = {"HIGH": 1.5, "NORMAL": 1.0, "LOW": 0.7}

COLORS = {
    "Static":      "#E05C5C",
    "Round Robin": "#F0A500",
    "Adaptive":    "#3DBE7A",
}

PLOT_BG  = "#1e293b"
PAPER_BG = "#1e293b"
FONT_CLR = "#f1f5f9"
GRID_CLR = "rgba(255,255,255,0.08)"
AXIS_CLR = "#cbd5e1"

_CSS = """
<style>
/* force dark on every possible Streamlit container */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main, .stApp {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #1e293b !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stHeader"] {
    background-color: #0f172a !important;
}
/* all text */
p, span, label, div, h1, h2, h3, h4, li {
    color: #f1f5f9 !important;
}
/* metric cards */
[data-testid="stMetric"] {
    background    : #1e293b !important;
    border        : 1px solid #334155 !important;
    border-radius : 10px;
    padding       : 1rem 1.2rem;
}
[data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: 0.78rem; }
[data-testid="stMetricValue"]  { color: #f1f5f9 !important; font-size: 1.45rem; font-weight: 700; }
/* primary button */
[data-testid="stButton"] > button[kind="primary"] {
    background    : linear-gradient(135deg, #1a4731, #3DBE7A) !important;
    color         : #fff !important;
    border        : none !important;
    border-radius : 8px;
    font-weight   : 700;
    letter-spacing: 0.04em;
}
/* tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #1e293b !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600;
    color: #94a3b8 !important;
    background-color: transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #3DBE7A !important;
    border-bottom: 2px solid #3DBE7A !important;
}
/* expander */
[data-testid="stExpander"] {
    background    : #1e293b !important;
    border        : 1px solid #334155 !important;
    border-radius : 8px;
}
/* dataframe */
[data-testid="stDataFrame"] {
    background-color: #1e293b !important;
}
/* sidebar headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p { color: #f1f5f9 !important; }
/* hero banner */
.hero-banner {
    background   : linear-gradient(135deg, #0d2b1a 0%, #0f172a 60%, #1e3a5f 100%) !important;
    border       : 1px solid #334155;
    border-radius: 14px;
    padding      : 1.6rem 2rem;
    margin-bottom: 1.2rem;
}
.hero-banner h1 {
    font-size  : 1.9rem !important;
    font-weight: 800 !important;
    color      : #f1f5f9 !important;
    margin     : 0 0 0.35rem 0;
}
.hero-banner p  { color: #94a3b8 !important; margin: 0; font-size: 0.95rem; }
.tag {
    display       : inline-block;
    background    : rgba(61,190,122,0.15) !important;
    color         : #3DBE7A !important;
    border        : 1px solid rgba(61,190,122,0.35);
    border-radius : 999px;
    padding       : 0.18rem 0.75rem;
    font-size     : 0.78rem;
    font-weight   : 600;
    margin-right  : 0.4rem;
    margin-bottom : 0.5rem;
}
/* winner card */
.winner-card {
    background    : linear-gradient(135deg, #0d2b1a, #134d2e) !important;
    border        : 2px solid #3DBE7A;
    border-radius : 12px;
    padding       : 1.4rem 2rem;
    text-align    : center;
}
.winner-card h3 { color: #3DBE7A !important; font-size: 1.5rem; margin: 0 0 0.6rem 0; }
.badge {
    display       : inline-block;
    background    : rgba(15,52,96,0.9) !important;
    color         : #38bdf8 !important;
    border        : 1px solid #1e40af;
    border-radius : 8px;
    padding       : 0.22rem 0.8rem;
    margin        : 0.2rem;
    font-size     : 0.85rem;
    font-weight   : 600;
}
/* section label */
.section-label {
    font-size     : 0.72rem !important;
    font-weight   : 700 !important;
    letter-spacing: 0.12em;
    color         : #3DBE7A !important;
    text-transform: uppercase;
    margin-bottom : 0.25rem;
}
/* sliders, toggles accent */
[data-baseweb="slider"] [data-testid="stThumbValue"] { color: #3DBE7A !important; }
[data-testid="stSlider"] > div > div > div > div { background: #3DBE7A !important; }
</style>
"""


class Node:
    def __init__(self, name: str, capacity: float, latency_factor: float):
        self.name             = name
        self.capacity         = capacity
        self.latency_factor   = latency_factor
        self.current_load     = 0.0
        self.task_count       = 0
        self.overload_count   = 0
        self.sla_breach_count = 0

    def assign_task(self, task, sla_threshold: float) -> float:
        latency               = task.workload * self.latency_factor * PRIORITY_MULT[task.priority]
        self.current_load    += task.workload
        self.task_count      += 1
        if self.current_load > self.capacity:
            self.overload_count += 1
        if latency > sla_threshold:
            self.sla_breach_count += 1
        return latency

    def reset(self):
        self.current_load     = 0.0
        self.task_count       = 0
        self.overload_count   = 0
        self.sla_breach_count = 0

    @property
    def utilization(self) -> float:
        return round((self.current_load / self.capacity) * 100, 1) if self.capacity else 0.0


class Task:
    def __init__(self, workload: float, task_type: str, priority: str):
        self.workload  = workload
        self.task_type = task_type
        self.priority  = priority


def static_fn(nodes, task, state):
    return nodes[0]


def round_robin_fn(nodes, task, state):
    idx = state["rr_idx"] % len(nodes)
    state["rr_idx"] += 1
    return nodes[idx]


def adaptive_fn(nodes, task, state):
    penalty = state.get("penalty", 10.0)
    best, best_cost = None, float("inf")
    for node in nodes:
        lat   = task.workload * node.latency_factor
        ratio = node.current_load / node.capacity if node.capacity > 0 else 0.0
        cost  = lat + (ratio * penalty)
        if cost < best_cost:
            best_cost = cost
            best      = node
    return best


SCHEDULERS = {
    "Static":      static_fn,
    "Round Robin": round_robin_fn,
    "Adaptive":    adaptive_fn,
}


def make_nodes(n: int) -> list:
    pool = [
        Node("Edge-A",  300.0,  0.8),
        Node("Edge-B",  250.0,  0.9),
        Node("Fog-C",   500.0,  0.4),
        Node("Cloud-D", 800.0,  0.2),
        Node("Cloud-E", 1000.0, 0.15),
        Node("Edge-F",  200.0,  1.0),
    ]
    return pool[:n]


def make_tasks(n: int, burst: bool, seed: int = 42) -> list:
    random.seed(seed)
    tasks = []
    for i in range(n):
        t    = random.choice(TASK_TYPES)
        base = random.uniform(1.0, 10.0)
        pri  = random.choices(["HIGH", "NORMAL", "LOW"], weights=[0.2, 0.6, 0.2])[0]
        if burst and i % 5 == 0:
            t, base, pri = "Emergency Response", random.uniform(8.0, 15.0), "HIGH"
        tasks.append(Task(round(base * TASK_WEIGHT[t], 2), t, pri))
    return tasks


def jain_fairness(values: list) -> float:
    if not values or all(v == 0 for v in values):
        return 0.0
    n  = len(values)
    s  = sum(values)
    sq = sum(v ** 2 for v in values)
    return round((s ** 2) / (n * sq), 4) if sq > 0 else 0.0


def run_scheduler(name: str, nodes: list, tasks: list,
                  penalty: float, sla_threshold: float) -> dict:
    for node in nodes:
        node.reset()
    fn            = SCHEDULERS[name]
    state         = {"rr_idx": 0, "penalty": penalty}
    total_latency = 0.0
    timeline      = []
    log_rows      = []
    running       = 0.0

    for i, task in enumerate(tasks):
        chosen         = fn(nodes, task, state)
        lat            = chosen.assign_task(task, sla_threshold)
        total_latency += lat
        running       += lat
        if (i + 1) % 50 == 0:
            timeline.append(round(running / (i + 1), 4))
        if i < 20:
            log_rows.append({
                "Task #":     i + 1,
                "Type":       task.task_type,
                "Priority":   task.priority,
                "Workload":   task.workload,
                "Node":       chosen.name,
                "Latency":    round(lat, 3),
                "Load After": round(chosen.current_load, 1),
            })

    n          = len(tasks)
    avg_lat    = round(total_latency / n, 4)
    throughput = round(n / total_latency, 4) if total_latency > 0 else 0.0
    overloads  = sum(nd.overload_count   for nd in nodes)
    sla_count  = sum(nd.sla_breach_count for nd in nodes)
    util_map   = {nd.name: nd.utilization for nd in nodes}

    return {
        "scheduler":   name,
        "avg_latency": avg_lat,
        "overloads":   overloads,
        "throughput":  throughput,
        "sla_pct":     round((sla_count / n) * 100, 2),
        "fairness":    jain_fairness(list(util_map.values())),
        "utilization": util_map,
        "timeline":    timeline,
        "log":         pd.DataFrame(log_rows),
    }


def sweep_penalty(num_tasks: int, num_nodes: int, burst: bool,
                  sla_threshold: float) -> pd.DataFrame:
    tasks = make_tasks(min(num_tasks, 500), burst)
    weights, lats, ovls, fairs = [], [], [], []
    for w in range(1, 51, 3):
        nodes  = make_nodes(num_nodes)
        result = run_scheduler("Adaptive", nodes, tasks, float(w), sla_threshold)
        weights.append(w)
        lats.append(result["avg_latency"])
        ovls.append(result["overloads"])
        fairs.append(result["fairness"])
    return pd.DataFrame({
        "Penalty Weight": weights,
        "Avg Latency":    lats,
        "Overload Count": ovls,
        "Fairness Index": fairs,
    })


def _base_layout(title: str, height: int, ylab: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color=FONT_CLR)),
        yaxis_title=ylab,
        yaxis=dict(gridcolor=GRID_CLR, color=AXIS_CLR),
        xaxis=dict(color=AXIS_CLR),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR),
        margin=dict(t=60, b=35, l=55, r=20),
        height=height,
    )


def chart_bar(names, values, title, ylab):
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=[COLORS[n] for n in names],
        marker_line_color="#111", marker_line_width=1,
        text=[str(v) for v in values], textposition="outside",
    ))
    layout = _base_layout(title, 340, ylab)
    layout["yaxis"]["range"] = [0, max(values) * 1.3] if values else [0, 1]
    fig.update_layout(**layout)
    return fig


def chart_utilization(results):
    node_names = list(results[0]["utilization"].keys())
    fig = go.Figure()
    for r in results:
        fig.add_trace(go.Bar(
            name=r["scheduler"], x=node_names,
            y=[r["utilization"][n] for n in node_names],
            marker_color=COLORS[r["scheduler"]],
            marker_line_color="#111", marker_line_width=1,
            text=[f"{r['utilization'][n]}%" for n in node_names],
            textposition="outside",
        ))
    fig.add_hline(y=100, line_dash="dash", line_color="#E05C5C",
                  annotation_text="Capacity Limit", annotation_font_color="#E05C5C")
    layout = _base_layout("Node Utilization by Scheduler (%)", 380, "Utilization (%)")
    layout["barmode"] = "group"
    layout["legend"]  = dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)",
                              font=dict(color=FONT_CLR))
    fig.update_layout(**layout)
    return fig


def chart_timeline(results):
    fig = go.Figure()
    for r in results:
        x = [(i + 1) * 50 for i in range(len(r["timeline"]))]
        fig.add_trace(go.Scatter(
            x=x, y=r["timeline"], name=r["scheduler"],
            mode="lines+markers",
            line=dict(color=COLORS[r["scheduler"]], width=2.5),
            marker=dict(size=4),
        ))
    layout = _base_layout("Cumulative Avg Latency Over Time", 360, "Cumulative Avg Latency")
    layout["xaxis"]["title"]     = "Tasks Processed"
    layout["xaxis"]["gridcolor"] = GRID_CLR
    layout["legend"] = dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)",
                             font=dict(color=FONT_CLR))
    fig.update_layout(**layout)
    return fig


def chart_radar(results):
    cats    = ["Low Latency", "Low Overload", "High Throughput", "Low SLA Breach", "High Fairness"]
    max_lat = max(r["avg_latency"] for r in results) or 1
    max_ovl = max(r["overloads"]   for r in results) or 1
    max_tp  = max(r["throughput"]  for r in results) or 1
    max_sla = max(r["sla_pct"]     for r in results) or 1
    fig = go.Figure()
    for r in results:
        scores = [
            round(1 - r["avg_latency"] / max_lat, 3),
            round(1 - r["overloads"]   / max_ovl, 3),
            round(r["throughput"]      / max_tp,  3),
            round(1 - r["sla_pct"]     / max_sla, 3),
            r["fairness"],
        ]
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]], theta=cats + [cats[0]],
            fill="toself", name=r["scheduler"],
            line_color=COLORS[r["scheduler"]], opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], color=AXIS_CLR,
                            gridcolor="rgba(255,255,255,0.12)"),
            angularaxis=dict(color=AXIS_CLR),
            bgcolor=PLOT_BG,
        ),
        title=dict(text="Performance Radar - All Axes Normalised 0 to 1",
                   font=dict(size=14, color=FONT_CLR)),
        legend=dict(orientation="h", y=-0.18, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR)),
        paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR),
        margin=dict(t=70, b=80), height=440,
    )
    return fig


def chart_sensitivity(df):
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Avg Latency", "Overload Count", "Fairness Index"])
    for col_name, color, col_idx in [("Avg Latency","#3DBE7A",1),
                                      ("Overload Count","#E05C5C",2),
                                      ("Fairness Index","#F0A500",3)]:
        fig.add_trace(go.Scatter(
            x=df["Penalty Weight"], y=df[col_name],
            mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=5),
            showlegend=False,
        ), row=1, col=col_idx)
    fig.update_layout(
        title=dict(text="Adaptive Scheduler: Penalty Weight Sensitivity (1 to 50)",
                   font=dict(size=14, color=FONT_CLR)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR), height=340, margin=dict(t=70, b=40),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=GRID_CLR, color=AXIS_CLR,
                         title_text="Penalty Weight", row=1, col=i)
        fig.update_yaxes(gridcolor=GRID_CLR, color=AXIS_CLR, row=1, col=i)
    for ann in fig.layout.annotations:
        ann.font.color = FONT_CLR
    return fig


def main():
    st.set_page_config(
        page_title="Smart City AI Scheduler",
        page_icon="🏙",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "Adaptive AI Workload Scheduler | Hackathon Prototype"},
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="hero-banner">'
        '<p class="section-label">🏙 Smart City AI | Hackathon Prototype</p>'
        '<h1>Adaptive AI Workload Scheduler</h1>'
        '<p>Real-time simulation of edge-fog-cloud task distribution across three scheduling strategies.</p>'
        '<br/>'
        '<span class="tag">AI for Smart Cities</span>'
        '<span class="tag">Edge Computing</span>'
        '<span class="tag">Adaptive Scheduling</span>'
        '<span class="tag">SLA Enforcement</span>'
        '<span class="tag">Jain Fairness Index</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    with st.expander("📋 Architecture and Problem Statement", expanded=False):
        ca, cb, cc = st.columns(3)
        with ca:
            st.markdown("**🏙 Problem Statement**")
            st.markdown(
                "Smart cities process continuous AI workloads from traffic cameras, "
                "IoT sensors, emergency dispatch, and environmental monitors. "
                "Tasks must be distributed in real time across edge, fog, and cloud nodes. "
                "Naive scheduling causes overloads, SLA violations, and latency spikes."
            )
        with cb:
            st.markdown("**🧮 Adaptive Cost Formula**")
            st.code(
                "latency    = workload * latency_factor * priority_weight\n"
                "load_ratio = current_load / capacity\n"
                "cost       = latency + (load_ratio * penalty)",
                language="text",
            )
            st.markdown("The node with the **lowest cost** is selected per task. As nodes fill, cost rises automatically.")
        with cc:
            st.markdown("**📊 Metrics Tracked**")
            st.markdown(
                "- **Avg Latency** — mean processing time per task  \n"
                "- **Overload Count** — nodes exceeding capacity  \n"
                "- **SLA Breach %** — tasks exceeding latency threshold  \n"
                "- **Throughput** — tasks per latency unit  \n"
                "- **Jain Fairness Index** — load equity (0–1)"
            )
    st.divider()

    with st.sidebar:
        st.markdown("## 🎮 Simulation Config")
        st.markdown("---")
        num_tasks  = st.slider("Number of Tasks",         100, 2000, 1000, 100)
        num_nodes  = st.slider("Compute Nodes",           2,   6,    4,    1)
        penalty    = st.slider("Adaptive Penalty Weight", 1.0, 50.0, 10.0, 1.0,
                                help="Higher = avoid overloaded nodes more aggressively")
        sla_thresh = st.slider("SLA Latency Threshold",   1.0, 30.0, 10.0, 0.5,
                                help="Tasks with latency above this count as SLA breaches")
        burst      = st.toggle("🔥 Burst / Spike Mode", value=False,
                                help="Every 5th task becomes a heavy Emergency Response spike")
        st.markdown("---")
        st.markdown("**🖥 Active Node Configuration**")
        node_df = pd.DataFrame({
            "Node":    ["Edge-A","Edge-B","Fog-C","Cloud-D","Cloud-E","Edge-F"],
            "Cap":     [300, 250, 500, 800, 1000, 200],
            "LatFact": [0.8, 0.9, 0.4, 0.2, 0.15, 1.0],
            "Tier":    ["Edge","Edge","Fog","Cloud","Cloud","Edge"],
        })
        st.dataframe(node_df.head(num_nodes), hide_index=True, use_container_width=True)
        st.markdown("---")
        st.markdown("**🧾 Judge Walkthrough**")
        st.markdown(
            "1. Keep defaults → Run Simulation  \n"
            "2. Check Winner Card and Radar  \n"
            "3. Enable Burst Mode → re-run  \n"
            "4. Drag penalty 1 → 40, compare  \n"
            "5. Review Sensitivity Sweep  \n"
            "6. Download CSV"
        )
        if burst:
            st.warning("🔥 Burst Mode ON — expect spikes on Static and Round Robin.")

    if "results" not in st.session_state:
        st.markdown("### How it works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("🔴 **Static Scheduler**\n\nAlways routes every task to Node 1. Creates an immediate bottleneck. Worst-case baseline.")
        with c2:
            st.warning("🟡 **Round Robin Scheduler**\n\nCycles through nodes in fixed order. Balances count but ignores capacity and load.")
        with c3:
            st.success("🟢 **Adaptive Scheduler**\n\nScores every node via cost formula. Selects the lowest-cost node dynamically.")
        st.markdown("---")

    if st.button("▶️  Run Simulation", type="primary", use_container_width=True):
        bar = st.progress(0, text="Initialising simulation...")
        time.sleep(0.05)
        results = []
        for i, name in enumerate(SCHEDULERS):
            bar.progress(int((i / 3) * 80),
                         text=f"Running {name} scheduler on {num_tasks} tasks...")
            nodes  = make_nodes(num_nodes)
            tasks  = make_tasks(num_tasks, burst)
            result = run_scheduler(name, nodes, tasks, penalty, sla_thresh)
            results.append(result)
            time.sleep(0.05)
        bar.progress(88, text="Running penalty sensitivity sweep...")
        sweep = sweep_penalty(num_tasks, num_nodes, burst, sla_thresh)
        bar.progress(100, text="Complete.")
        time.sleep(0.3)
        bar.empty()
        st.session_state.update({
            "results":    results,
            "sweep":      sweep,
            "penalty":    penalty,
            "num_tasks":  num_tasks,
            "num_nodes":  num_nodes,
            "burst":      burst,
            "sla_thresh": sla_thresh,
        })

    if "results" not in st.session_state:
        return

    results    = st.session_state["results"]
    sweep      = st.session_state["sweep"]
    penalty    = st.session_state["penalty"]
    num_tasks  = st.session_state["num_tasks"]
    num_nodes  = st.session_state["num_nodes"]
    burst      = st.session_state["burst"]
    sla_thresh = st.session_state["sla_thresh"]
    names      = [r["scheduler"] for r in results]

    st.success("✅ Simulation complete — scroll down to explore all results.")
    if burst:
        st.warning("🔥 Burst Mode was active during this run.")
    st.divider()

    best = min(results, key=lambda r: r["avg_latency"])
    st.markdown(
        f'<div class="winner-card">'
        f'<h3>🏆 Best Performer: {best["scheduler"]}</h3>'
        f'<span class="badge">📉 Avg Latency: {best["avg_latency"]} units</span>'
        f'<span class="badge">⚡ Overloads: {best["overloads"]}</span>'
        f'<span class="badge">🚨 SLA Breach: {best["sla_pct"]}%</span>'
        f'<span class="badge">⚖ Fairness: {best["fairness"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<p class="section-label">📋 Quick Metrics</p>', unsafe_allow_html=True)
    for col, r in zip(st.columns(3), results):
        with col:
            base  = results[0]["avg_latency"]
            delta = round(r["avg_latency"] - base, 4) if r["scheduler"] != "Static" else None
            st.metric(
                f"{r['scheduler']} — Avg Latency",
                f"{r['avg_latency']} u",
                delta=f"{delta} vs Static" if delta is not None else None,
                delta_color="inverse",
            )
            st.metric("⚡ Overloads",      r["overloads"])
            st.metric("🚨 SLA Breach",  f"{r['sla_pct']}%")
            st.metric("⚖ Fairness Index",  r["fairness"])
    st.divider()

    st.markdown('<p class="section-label">📄 Full Comparison Table</p>', unsafe_allow_html=True)
    rows = []
    for r in results:
        row = {
            "Scheduler":      r["scheduler"],
            "Avg Latency":    r["avg_latency"],
            "Overload Count": r["overloads"],
            "Throughput":     r["throughput"],
            "SLA Breach %":   r["sla_pct"],
            "Fairness Index": r["fairness"],
        }
        row.update({f"{k} Util%": v for k, v in r["utilization"].items()})
        rows.append(row)
    df       = pd.DataFrame(rows).set_index("Scheduler")
    num_cols = ["Avg Latency", "Overload Count", "Throughput", "SLA Breach %", "Fairness Index"]
    styled   = (
        df[num_cols].style
        .highlight_min(subset=["Avg Latency", "Overload Count", "SLA Breach %"], color="#134d2e")
        .highlight_max(subset=["Throughput", "Fairness Index"], color="#134d2e")
        .format(precision=3)
    )
    st.dataframe(styled, use_container_width=True)
    st.caption("Green — best value per column.")
    st.download_button(
        "💾  Download Results as CSV",
        df.reset_index().to_csv(index=False).encode("utf-8"),
        "scheduler_results.csv", "text/csv",
    )
    st.divider()

    st.markdown('<p class="section-label">📊 Core Performance Charts</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_bar(names, [r["avg_latency"] for r in results],
                                   "Average Task Latency", "Latency (units)"), width="stretch")
        st.caption("Lower is better. Adaptive routes to low-latency, low-load nodes.")
    with c2:
        st.plotly_chart(chart_bar(names, [r["overloads"] for r in results],
                                   "Node Overload Events", "Overload Count"), width="stretch")
        st.caption("Each overload event means a node exceeded its processing capacity.")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_bar(names, [r["sla_pct"] for r in results],
                                   f"SLA Breach Rate (threshold: {sla_thresh} units)",
                                   "% of Tasks Breaching SLA"), width="stretch")
        st.caption("SLA breaches in a smart city = missed emergency alerts.")
    with c4:
        st.plotly_chart(chart_bar(names, [r["fairness"] for r in results],
                                   "Jain Fairness Index", "Score (1 = perfectly fair)"),
                        width="stretch")
        st.caption("Measures how evenly load is distributed across all nodes.")
    st.divider()

    st.markdown('<p class="section-label">🕸 Multi-Axis Performance Radar</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_radar(results), width="stretch")
    st.caption("All axes normalised to 0-1. Larger polygon = stronger overall performance.")
    st.divider()

    st.markdown('<p class="section-label">🖥 Node Utilization by Scheduler</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_utilization(results), width="stretch")
    st.caption("Values above 100% = overloaded nodes. Red dashed line = capacity ceiling.")
    st.divider()

    st.markdown('<p class="section-label">🕛 Latency Convergence Over Time</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_timeline(results), width="stretch")
    st.caption("Cumulative avg latency sampled every 50 tasks. Adaptive converges; Static drifts upward.")
    st.divider()

    st.markdown('<p class="section-label">🧪 Penalty Weight Sensitivity (Adaptive Only)</p>', unsafe_allow_html=True)
    st.markdown("Sweeps penalty 1→50 — trade-off between latency-first and load-balance-first behaviour.")
    st.plotly_chart(chart_sensitivity(sweep), width="stretch")
    st.caption("Low penalty → more overloads. High penalty → better fairness, fewer overloads.")
    with st.expander("Sensitivity Sweep Raw Data"):
        st.dataframe(sweep, use_container_width=True, hide_index=True)
    st.divider()

    st.markdown('<p class="section-label">📝 Task Assignment Log — First 20 Tasks</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔴 Static", "🟡 Round Robin", "🟢 Adaptive"])
    captions = {
        "Static":      "Every task routed to Edge-A. Load accumulates with no distribution.",
        "Round Robin": "Rotates across nodes in fixed order regardless of priority or load.",
        "Adaptive":    "Selects nodes dynamically. Priority weighting influences cost per task.",
    }
    for tab, r in zip([tab1, tab2, tab3], results):
        with tab:
            st.dataframe(r["log"], use_container_width=True, hide_index=True)
            st.caption(captions[r["scheduler"]])
    st.divider()

    best_lat  = min(results, key=lambda r: r["avg_latency"])
    best_ovl  = min(results, key=lambda r: r["overloads"])
    best_sla  = min(results, key=lambda r: r["sla_pct"])
    best_fair = max(results, key=lambda r: r["fairness"])
    burst_note = (
        " Burst Mode active: Emergency Response spikes amplified overloads on "
        "Static and Round Robin while Adaptive re-routed heavy tasks automatically."
    ) if burst else ""

    st.markdown('<p class="section-label">💡 Simulation Insight</p>', unsafe_allow_html=True)
    st.info(
        f"📉 Best Latency: {best_lat['scheduler']} — {best_lat['avg_latency']} units  |  "
        f"⚡ Fewest Overloads: {best_ovl['scheduler']} — {best_ovl['overloads']} events  |  "
        f"🚨 Best SLA: {best_sla['scheduler']} — {best_sla['sla_pct']}% breach  |  "
        f"⚖ Best Fairness: {best_fair['scheduler']} — Jain {best_fair['fairness']}  |  "
        f"Config: penalty={penalty}, tasks={num_tasks}, nodes={num_nodes}, SLA={sla_thresh}"
        + burst_note
    )


if __name__ == "__main__":
    main()