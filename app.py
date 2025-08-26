# app.py — Lab Experiment Planner (Streamlit)
# Features:
# - Weekly/Monthly 9–5 scheduler with dependency handling
# - Template or free-text (LLM) parsing
# - Reagent calculator (C1V1=C2V2)
# - Protocol fetcher from local JSON
# - Notebook summary export
# - OpenAI-powered parsing + optimization suggestions (optional via OPENAI_API_KEY)

import os
import json
import re
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import streamlit as st

# -----------------------------
# OpenAI client (optional)
# -----------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

client = None
if OpenAI and os.getenv("OPENAI_API_KEY"):
    try:
        client = OpenAI()
    except Exception:
        client = None

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Lab Experiment Planner", page_icon="🧪", layout="wide")
st.title("🧪 Lab Experiment Planner — 9–5 Smart Scheduler")

# -----------------------------
# Load protocols.json (with safe fallback)
# -----------------------------
DEFAULT_PROTOCOLS = {
    "cell_timecourse": {
        "title": "Cell Treatment Timecourse (Generic)",
        "steps": [
            "Warm media and reagents to 37°C.",
            "Thaw cells if needed and check viability.",
            "Seed cells at desired density; allow to settle.",
            "Prepare drug dilutions from stock using sterile technique.",
            "Treat cells at T0; record exact time.",
            "At each timepoint: wash, lyse, and collect samples on ice.",
            "Store lysates at -20°C/-80°C or proceed to downstream assay."
        ]
    },
    "western_blot": {
        "title": "Western Blot – Rapid Check",
        "steps": [
            "Prepare lysates; quantify protein concentration.",
            "Cast/assemble gel; pre-run if required.",
            "Load samples with loading dye; run until proper separation.",
            "Transfer to membrane; confirm transfer (Ponceau).",
            "Block membrane; incubate in primary antibody.",
            "Wash, incubate in secondary antibody; wash again.",
            "Develop and document images; export to ELN."
        ]
    },
    "qpcr": {
        "title": "qPCR Workflow",
        "steps": [
            "Extract total RNA; assess purity (A260/A280).",
            "DNase treat and cleanup.",
            "Reverse transcribe to cDNA.",
            "Prepare qPCR master mix and plate layout.",
            "Run qPCR and collect amplification data.",
            "Perform ΔΔCt analysis; visualize results."
        ]
    },
    "elisa": {
        "title": "ELISA – Sandwich Format",
        "steps": [
            "Coat capture antibody; incubate.",
            "Block plate; wash.",
            "Add samples and standards; incubate.",
            "Wash; add detection antibody; incubate.",
            "Add substrate; stop reaction; read absorbance.",
            "Export data; calculate concentrations from standard curve."
        ]
    }
}

PROTOCOLS = DEFAULT_PROTOCOLS
try:
    with open("protocols.json", "r", encoding="utf-8") as f:
        PROTOCOLS = json.load(f)
except FileNotFoundError:
    st.warning("`protocols.json` not found — using built-in defaults. Add a protocols.json next to app.py to customize.")
except Exception as e:
    st.warning(f"Could not read protocols.json ({e}); using built-in defaults.")

# -----------------------------
# Core models & helpers
# -----------------------------
@dataclass
class Task:
    name: str
    duration_min: int
    earliest_start: dt.time
    latest_end: dt.time
    depends_on: List[str]
    fixed: bool = False
    preferred_start: Optional[dt.time] = None

def make_template(exptype: str) -> List[Task]:
    exptype = exptype.lower()
    if exptype == "cell_timecourse":
        return [
            Task("Thaw cells & warm media", 20, dt.time(9,0), dt.time(10,30), []),
            Task("Seed cells (3 doses × 3 reps)", 40, dt.time(9,20), dt.time(11,30), ["Thaw cells & warm media"]),
            Task("Incubation settling (post-seed)", 30, dt.time(10,0), dt.time(12,0), ["Seed cells (3 doses × 3 reps)"]),
            Task("Prepare drug dilutions", 30, dt.time(10,30), dt.time(13,0), ["Incubation settling (post-seed)"]),
            Task("Treat cells (T0)", 10, dt.time(11,0), dt.time(13,0), ["Prepare drug dilutions"], fixed=True, preferred_start=dt.time(11,30)),
            Task("1-hour TP: wash/lyse", 25, dt.time(12,15), dt.time(14,0), ["Treat cells (T0)"]),
            Task("4-hour TP: wash/lyse", 30, dt.time(15,15), dt.time(16,30), ["Treat cells (T0)"]),
            Task("Prep mini-gel & buffers", 35, dt.time(12,0), dt.time(16,30), ["1-hour TP: wash/lyse"]),
            Task("Run mini-gel (test lane)", 40, dt.time(13,0), dt.time(16,45), ["Prep mini-gel & buffers"]),
            Task("Cleanup & notes", 20, dt.time(16,30), dt.time(17,0), ["4-hour TP: wash/lyse","Run mini-gel (test lane)"]),
        ]
    if exptype == "western_blot":
        return [
            Task("Lysate prep & quant", 45, dt.time(9,0), dt.time(12,0), []),
            Task("Gel casting/prep", 35, dt.time(9,30), dt.time(12,30), []),
            Task("Run gel", 60, dt.time(11,0), dt.time(15,0), ["Lysate prep & quant","Gel casting/prep"]),
            Task("Transfer", 60, dt.time(12,30), dt.time(16,30), ["Run gel"]),
            Task("Block membrane", 60, dt.time(13,30), dt.time(17,0), ["Transfer"]),
            Task("Primary antibody O/N", 10, dt.time(16,45), dt.time(17,0), ["Block membrane"], fixed=True, preferred_start=dt.time(16,50)),
        ]
    if exptype == "qpcr":
        return [
            Task("RNA extraction", 60, dt.time(9,0), dt.time(12,0), []),
            Task("DNase & cleanup", 30, dt.time(10,0), dt.time(13,0), ["RNA extraction"]),
            Task("cDNA synthesis", 45, dt.time(11,0), dt.time(14,0), ["DNase & cleanup"]),
            Task("qPCR plate setup", 40, dt.time(12,0), dt.time(16,0), ["cDNA synthesis"]),
            Task("Run qPCR", 60, dt.time(13,0), dt.time(17,0), ["qPCR plate setup"]),
            Task("Export & quick analysis", 30, dt.time(14,0), dt.time(17,0), ["Run qPCR"]),
        ]
    if exptype == "elisa":
        return [
            Task("Plate coating (if needed)", 60, dt.time(9,0), dt.time(12,0), []),
            Task("Blocking", 60, dt.time(10,30), dt.time(14,0), ["Plate coating (if needed)"]),
            Task("Add samples/standards", 30, dt.time(12,0), dt.time(15,0), ["Blocking"]),
            Task("Incubation & washes", 60, dt.time(12,30), dt.time(16,0), ["Add samples/standards"]),
            Task("Substrate & read", 30, dt.time(13,30), dt.time(17,0), ["Incubation & washes"]),
            Task("Export & quick analysis", 30, dt.time(14,0), dt.time(17,0), ["Substrate & read"]),
        ]
    # default
    return make_template("cell_timecourse")

def parse_paragraph_to_template(text: str) -> List[Task]:
    # lightweight regex fallback when no LLM
    t = text.lower()
    doses = re.findall(r'(\d+\.?\d*)\s*(n?m|u?m|mm|m)\b', t)
    tps  = re.findall(r'(\d+)\s*h\b', t)
    wants_gel = ("gel" in t) or ("western" in t)
    wants_seed = ("seed" in t) or ("plating" in t)

    tasks: List[Task] = []
    if wants_seed:
        tasks += [
            Task("Thaw cells & warm media", 20, dt.time(9,0), dt.time(11,0), []),
            Task("Seed cells", 40, dt.time(9,20), dt.time(12,0), ["Thaw cells & warm media"]),
            Task("Incubation settling (post-seed)", 30, dt.time(10,0), dt.time(13,0), ["Seed cells"]),
        ]
    if doses:
        tasks.append(Task("Prepare drug dilutions", 30, dt.time(10,0), dt.time(14,0), [] if not wants_seed else ["Incubation settling (post-seed)"]))
        tasks.append(Task("Treat cells (T0)", 10, dt.time(11,0), dt.time(14,0), ["Prepare drug dilutions"], fixed=True, preferred_start=dt.time(11,30)))
    dep_after_treat = []
    if tps:
        sorted_tps = sorted([int(x) for x in tps[:2]])
        for hrs in sorted_tps:
            tp_name = f"{hrs}-hour TP: wash/lyse"
            tasks.append(Task(tp_name, 30, dt.time(9,0), dt.time(17,0), ["Treat cells (T0)"]))
            dep_after_treat.append(tp_name)
    if wants_gel:
        tasks += [
            Task("Prep mini-gel & buffers", 35, dt.time(9,0), dt.time(16,30), dep_after_treat[:1] if dep_after_treat else []),
            Task("Run mini-gel (test lane)", 40, dt.time(11,0), dt.time(16,45), ["Prep mini-gel & buffers"]),
        ]
    # Cleanup depends on last 2 tasks if available
    deps_cleanup: List[str] = []
    for candidate in reversed(tasks):
        if candidate.name not in deps_cleanup:
            deps_cleanup.append(candidate.name)
        if len(deps_cleanup) >= 2:
            break
    tasks.append(Task("Cleanup & notes", 20, dt.time(16,0), dt.time(17,0), deps_cleanup))
    return tasks if tasks else make_template("cell_timecourse")

# -----------------------------
# OpenAI structured parsing schema + functions
# -----------------------------
# OpenAI tool spec (Responses API expects name at top level)
TOOL_SPEC = {
    "type": "function",
    "name": "extract_experiment",
    "description": "Extract experiment entities from a paragraph for lab planning.",
    "parameters": {
        "type": "object",
        "properties": {
            "assay": {"type":"string", "description":"cell_timecourse, western_blot, qpcr, or elisa"},
            "cell_line": {"type":"string"},
            "doses": {"type":"array","items":{"type":"string"}},
            "replicates":{"type":"integer"},
            "timepoints_hours":{"type":"array","items":{"type":"number"}},
            "needs_seeding":{"type":"boolean"},
            "include_gel_check":{"type":"boolean"},
            "other_notes":{"type":"string"}
        },
        "required": ["assay"]
    }
}

def llm_parse_paragraph_to_spec(text: str) -> Optional[dict]:
    if client is None:
        return None

    prompt = f"""
    You are a lab planner. Parse the following paragraph and return ONLY structured fields via the provided tool.
    If assay is unclear, choose the best fit from: cell_timecourse, western_blot, qpcr, elisa.
    Paragraph: ```{text}```
    """

    try:
        resp = client.responses.create(
            # Pick a model you have access to; these are reliable for structured outputs:
            # "gpt-4o-mini", "o3-mini", etc. If you were using "gpt-5" and don't have access,
            # the call will 404. Swap to one you know works in your account.
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": "Extract entities for lab scheduling."},
                {"role": "user", "content": prompt},
            ],
            tools=[TOOL_SPEC],
            tool_choice={"type": "function", "function": {"name": "extract_experiment"}},
            max_output_tokens=400,
        )

        # Pull function call arguments (SDKs may expose slightly different fields; cover both)
        for item in resp.output:
            t = getattr(item, "type", "")
            if t == "tool_call":
                # Some SDKs use 'name', others 'tool_name'
                tool_name = getattr(item, "name", None) or getattr(item, "tool_name", None)
                if tool_name == "extract_experiment":
                    args = getattr(item, "arguments", {})  # may already be a dict
                    if isinstance(args, str):
                        import json as _json
                        try:
                            args = _json.loads(args)
                        except Exception:
                            pass
                    if isinstance(args, dict):
                        return args
        return None

    except Exception as e:
        st.info(f"LLM parser unavailable ({e}); using fallback.")
        return None



def tasks_from_llm_spec(spec: dict) -> List[Task]:
    assay = (spec.get("assay") or "cell_timecourse").lower()
    if assay in ("cell_timecourse", "timecourse"):
        tasks = make_template("cell_timecourse")
        tps = spec.get("timepoints_hours") or []
        wanted = sorted(set(int(x) for x in tps if isinstance(x, (int, float))))
        # Remove default TP tasks, then add wanted
        tasks = [t for t in tasks if "TP:" not in t.name]
        after = "Treat cells (T0)"
        for h in wanted:
            tasks.append(Task(f"{h}-hour TP: wash/lyse", 30, dt.time(9,0), dt.time(17,0), [after]))
        if spec.get("include_gel_check") is False:
            tasks = [t for t in tasks if "gel" not in t.name.lower()]
    elif assay == "western_blot":
        tasks = make_template("western_blot")
    elif assay == "qpcr":
        tasks = make_template("qpcr")
    elif assay == "elisa":
        tasks = make_template("elisa")
    else:
        tasks = make_template("cell_timecourse")
    return tasks

# -----------------------------
# Scheduler, ICS, reagents, notebook
# -----------------------------
def schedule_multiday(tasks: List[Task], start_date: dt.date, num_days: int,
                      work_start=dt.time(9,0), work_end=dt.time(17,0)) -> pd.DataFrame:
    from collections import defaultdict, deque
    name_to_task = {t.name: t for t in tasks}
    graph = defaultdict(list)
    indegree = defaultdict(int)
    for t in tasks:
        for dep in t.depends_on:
            if dep in name_to_task:
                graph[dep].append(t.name)
                indegree[t.name] += 1
        indegree.setdefault(t.name, 0)
    q = deque([n for n in indegree if indegree[n]==0])
    order: List[str] = []
    while q:
        n = q.popleft()
        order.append(n)
        for nxt in graph[n]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    # Append any nodes missed (unmet deps) to avoid deadlock
    for n in name_to_task:
        if n not in order:
            order.append(n)

    class Scheduled:
        def __init__(self, task, start_dt, end_dt, day_index):
            self.task = task
            self.start_dt = start_dt
            self.end_dt = end_dt
            self.day_index = day_index

    scheduled: List[Scheduled] = []

    def latest_dep_end(task: Task):
        latest = None
        for s in scheduled:
            if s.task.name in task.depends_on:
                latest = s.end_dt if latest is None else max(latest, s.end_dt)
        return latest

    def clamp(day, tm):
        return dt.datetime.combine(day, tm)

    for name in order:
        t = name_to_task[name]
        placed = False
        for day_idx in range(num_days):
            day = start_date + dt.timedelta(days=day_idx)
            if day.weekday() >= 5:
                continue
            dstart = clamp(day, work_start)
            dend = clamp(day, work_end)
            if t.fixed and t.preferred_start:
                start_dt = clamp(day, t.preferred_start)
            else:
                dep_end = latest_dep_end(t)
                start_dt = dstart if dep_end is None else max(dstart, dep_end)
                start_dt = max(start_dt, clamp(day, t.earliest_start))
            end_dt = start_dt + dt.timedelta(minutes=t.duration_min)
            if end_dt > clamp(day, t.latest_end):
                continue
            # overlap check
            conflict = False
            for s in scheduled:
                if s.day_index == day_idx and not (end_dt <= s.start_dt or start_dt >= s.end_dt):
                    conflict = True
                    break
            if not conflict and end_dt <= dend:
                scheduled.append(Scheduled(t, start_dt, end_dt, day_idx))
                placed = True
                break
        if not placed:
            # best-effort drop on last day if possible
            last_day = start_date + dt.timedelta(days=num_days-1)
            dstart = clamp(last_day, work_start)
            dend = clamp(last_day, work_end)
            start_dt = max(dstart, clamp(last_day, t.earliest_start))
            end_dt = start_dt + dt.timedelta(minutes=t.duration_min)
            if end_dt <= dend:
                scheduled.append(Scheduled(t, start_dt, end_dt, num_days-1))
            else:
                st.warning(f"Could not place task: {t.name}")

    rows = []
    for s in scheduled:
        rows.append({
            "Task": s.task.name,
            "Start": s.start_dt,
            "End": s.end_dt,
            "Duration_min": int((s.end_dt - s.start_dt).total_seconds() // 60),
            "Day": (start_date + dt.timedelta(days=s.day_index)).isoformat(),
        })
    return pd.DataFrame(rows).sort_values(["Day", "Start"])

def build_ics(df: pd.DataFrame) -> str:
    def dtstamp(x: dt.datetime): return x.strftime("%Y%m%dT%H%M%S")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//LabAgent//Planner//EN"]
    for _, r in df.iterrows():
        lines += ["BEGIN:VEVENT",
                  f"DTSTART:{dtstamp(r['Start'])}",
                  f"DTEND:{dtstamp(r['End'])}",
                  f"SUMMARY:{r['Task']}",
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\n".join(lines)

def reagent_calculator(components: List[dict]) -> List[dict]:
    # components: [{"name":"Drug A","stock":"10 mM","final":"1 uM","final_volume_ml":10}, ...]
    def to_M(val, unit):
        unit = unit.lower().strip()
        factor = {"m":1, "mm":1e-3, "um":1e-6, "nm":1e-9}.get(unit)
        if factor is None:
            raise ValueError(f"Unknown unit {unit} (use M, mM, uM, nM)")
        return float(val) * factor
    results = []
    for comp in components:
        s_val, s_unit = comp["stock"].split()
        f_val, f_unit = comp["final"].split()
        C1 = to_M(s_val, s_unit)
        C2 = to_M(f_val, f_unit)
        V2_L = float(comp["final_volume_ml"]) / 1000.0
        V1_L = (C2 * V2_L) / C1
        V1_uL = V1_L * 1e6
        V2_uL = float(comp["final_volume_ml"]) * 1000.0
        results.append({
            "name": comp["name"],
            "stock": comp["stock"],
            "target": f"{comp['final']} in {comp['final_volume_ml']} mL",
            "add_stock_uL": round(V1_uL, 2),
            "add_solvent_uL": round(V2_uL - round(V1_uL, 2), 2),
        })
    return results

def notebook_summary(df: pd.DataFrame, experiment_name: str, protocol_key: str,
                     reagents: Optional[List[dict]] = None) -> str:
    steps = PROTOCOLS.get(protocol_key, {}).get("steps", [])
    lines = [f"# {experiment_name}", "", "## Auto-Generated Plan"]
    for d, sub in df.groupby("Day"):
        lines.append(f"### {d}")
        for _, r in sub.iterrows():
            lines.append(f"- {r['Start'].strftime('%H:%M')}–{r['End'].strftime('%H:%M')}: {r['Task']}")
    if reagents:
        lines.append("")
        lines.append("## Reagent Recipes")
        for r in reagents:
            lines.append(f"- {r['name']}: add {r['add_stock_uL']} µL stock, {r['add_solvent_uL']} µL solvent → {r['target']}")
    if steps:
        lines.append("")
        lines.append("## Protocol Snippet")
        for i, s in enumerate(steps, 1):
            lines.append(f"{i}. {s}")
    lines.append("")
    lines.append(f"_Generated on {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    return "\n".join(lines)

# -----------------------------
# UI: Planner + Tools
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Plan Settings")
    horizon = st.selectbox("Planning horizon", ["Weekly (next Mon–Fri)", "Monthly (~4 work weeks)"])
    mode = st.radio("Describe by…", ["Template type", "Paragraph description"])

    if mode == "Template type":
        exptype = st.selectbox("Experiment type", ["cell_timecourse", "western_blot", "qpcr", "elisa"])
        tasks = make_template(exptype)
        proto_key = exptype
    else:
        paragraph = st.text_area(
            "Describe your experiment",
            height=120,
            value="3 doses of cisplatin on SKOV3; 1h & 4h timepoints; quick mini-gel; seed cells."
        )
        spec = llm_parse_paragraph_to_spec(paragraph)
        if spec:
            tasks = tasks_from_llm_spec(spec)
            proto_key = (spec.get("assay") or "cell_timecourse").lower()
            st.caption("🧠 Parsed with OpenAI (structured).")
        else:
            tasks = parse_paragraph_to_template(paragraph)
            proto_key = "cell_timecourse"
            st.caption("ℹ️ Using fallback regex parser (no/failed API).")

    # schedule window
    today = dt.date.today()
    days_ahead = (0 - today.weekday() + 7) % 7
    if days_ahead == 0:
        # if Monday, start next Monday to avoid conflicting with current day
        days_ahead = 7
    start_date = today + dt.timedelta(days=days_ahead)
    num_days = 5 if "Weekly" in horizon else 20

    df = schedule_multiday(tasks, start_date, num_days)
    st.dataframe(df, use_container_width=True)

    # downloads
    ics_text = build_ics(df)
    st.download_button("📅 Download .ICS (all reminders)", data=ics_text, file_name="lab_plan.ics", mime="text/calendar")
    st.download_button("📄 Download CSV plan", data=df.to_csv(index=False), file_name="lab_plan.csv", mime="text/csv")

    # ---------- Optimization suggestions (place right after table + buttons)
    with st.expander("🧠 Optimization suggestions", expanded=False):
        if client and not df.empty:
            try:
                plan_txt = "\n".join(
                    f"{r.Start.strftime('%H:%M')}–{r.End.strftime('%H:%M')}: {r.Task} ({r.Day})"
                    for _, r in df.iterrows()
                )
                proto = PROTOCOLS.get(proto_key, {}).get("steps", [])
                prompt = f"""
                You are a senior lab manager. Given this 9–5 schedule and protocol snippet, propose concrete improvements:
                - Safe parallelization
                - Re-ordering around fixed timepoints
                - Idle-time filling
                - Buffer prep timing
                - Quick wins to increase hypotheses/run
                Keep it short, bullet-pointed, and conservative.

                Schedule:
                {plan_txt}

                Protocol steps:
                {proto}
                """
                resp = client.responses.create(model="gpt-5", input=[{"role":"user","content":prompt}], max_output_tokens=400)
                text_out = []
                for item in resp.output:
                    if getattr(item, "type", "") == "output_text":
                        text_out.append(item.text)
                st.write("\n".join(text_out).strip() or "No suggestions.")
            except Exception as e:
                st.info(f"Optimization unavailable ({e}).")
        else:
            st.info("Add OPENAI_API_KEY in Secrets to enable AI suggestions.")

with col2:
    st.subheader("Reagent Calculator (C1V1 = C2V2)")
    st.caption("Add components to compute stock and solvent volumes.")
    num = st.number_input("Number of components", min_value=1, max_value=10, value=2, step=1)
    comps = []
    for i in range(num):
        with st.expander(f"Component {i+1}", expanded=(i < 2)):
            name = st.text_input(f"Name {i+1}", value=f"Drug {i+1}")
            stock = st.text_input(f"Stock conc {i+1} (e.g., '10 mM')", value="10 mM")
            final = st.text_input(f"Final conc {i+1} (e.g., '1 uM')", value="1 uM")
            vol = st.number_input(f"Final volume {i+1} (mL)", min_value=0.1, max_value=1000.0, value=10.0, step=0.1)
            comps.append({"name": name, "stock": stock, "final": final, "final_volume_ml": vol})

    if st.button("Calculate volumes"):
        try:
            results = reagent_calculator(comps)
            st.success("Calculated volumes:")
            st.table(results)
        except Exception as e:
            st.error(str(e))

st.subheader("Notebook Summary")
exp_name = st.text_input("Experiment name/title", value="Dose-response + WB quick check")
if st.button("Generate notebook text"):
    try:
        reagents = reagent_calculator(comps)
    except Exception:
        reagents = None
    text = notebook_summary(df, exp_name, proto_key, reagents)
    st.code(text)
    st.download_button("⬇️ Download notebook.txt", data=text, file_name="notebook_summary.txt", mime="text/plain")

st.divider()
st.caption("Local protocol library is loaded from 'protocols.json'. Extend that file to add your own SOPs.")
