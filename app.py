# app.py
import html
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    ROLES, SKILLS,
    neighbors, explain_step, neighbors_with_added_skills,
    get_paths, augment_current_with_skills, role_similarity,
    weighted_jaccard, _ROLE2SK, role_skills,
    parse_resume, score_roles_by_resume, extract_resume_skills,
    get_learning_links
)

# ---------------- Page ----------------
st.set_page_config(page_title="· Career Path Planner", layout="wide")

# ---------------- Light (iOS-like) CSS ----------------
st.markdown(
    """
<style>
:root{
  --bg:#ffffff; --ink:#0f172a; --muted:#475569; --border:#e5e7eb;
  --soft:#f8fafc; --soft-2:#f1f5f9; --chip-ink:#111827;
  --chip-amber:#FEF3C7; --chip-amber-ink:#92400E;
  --chip-mint:#D1FAE5; --chip-mint-ink:#065F46;
  --chip-indigo:#E0E7FF; --chip-indigo-ink:#3730A3;
  --card-shadow:0 1px 10px rgba(16,24,40,.06);
}
html, body, .stApp { background:var(--bg); color:var(--ink); }

/* Centered fancy title */
.title-wrap { display:flex; justify-content:center; align-items:center; margin: 6px 0 10px; }
.fancy-title {
  font-weight:800; letter-spacing:.2px; text-align:center;
  font-size: clamp(28px, 3.4vw, 44px);
  background: linear-gradient(92deg, #6366f1, #22c55e, #06b6d4);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  text-shadow: 0 10px 26px rgba(99,102,241,.18);
}
.subtitle { text-align:center; color:var(--muted); margin:-6px 0 12px; }

/* iOS-like inputs & expander */
.stTextArea textarea, .stTextInput input {
  background: var(--soft); color: var(--ink); border:1px solid var(--border); border-radius:12px;
}
div[data-testid="stFileUploader"] > div {
  background: var(--soft); padding:.35rem .5rem; border:1px solid var(--border); border-radius:12px;
}
.streamlit-expanderHeader {
  background: var(--soft); color: var(--ink);
  border:1px solid var(--border); border-radius:12px; padding:.6rem .9rem; font-weight:600;
}
div.streamlit-expanderContent {
  background: var(--soft-2); color: var(--ink);
  border:1px solid var(--border); border-top:0; border-radius:0 0 12px 12px; padding:1rem;
}

/* shared chips / cards */
.path-card{ border:1px solid var(--border); border-radius:16px; background:var(--soft);
  padding:16px 18px; margin:18px 0; box-shadow:var(--card-shadow); }
.kpi{display:inline-flex;align-items:center; gap:.55rem; padding:.45rem .7rem;
  border-radius:12px; background:var(--soft-2); border:1px solid var(--border); font-size:13px;}
.section-sub{font-weight:700; margin:8px 0 4px; text-align:center;}
.skills-box{text-align:center; margin-top:2px;}
.badge{display:inline-flex; align-items:center; justify-content:center;
  padding:6px 12px; margin:6px; border-radius:999px; font-size:13px; font-weight:600; color:var(--chip-ink);}
.badge.strength{background:var(--chip-amber); color:var(--chip-amber-ink);}
.badge.unlock{background:var(--chip-mint); color:var(--chip-mint-ink);}
.badge.detect{background:var(--chip-indigo); color:var(--chip-indigo-ink);}

.hard-divider{ height: 28px; margin: 12px 0 24px; border-bottom: 3px solid var(--border); }

/* remove plotly fullscreen */
button[title="View fullscreen"]{ display:none; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Title ----------------
st.markdown('<div class="title-wrap"><div class="fancy-title">Career Path Planner</div></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explore role transitions, compare fits, and see the exact skills to grow next.</div>', unsafe_allow_html=True)

# ---------------- Helpers ----------------
def strength_label(avg: float) -> str:
    if avg >= 0.55:  return "Path Strength: ⭐⭐⭐⭐ Very strong"
    if avg >= 0.40:  return "Path Strength: ⭐⭐⭐ Strong"
    if avg >= 0.28:  return "Path Strength: ⭐⭐ Moderate"
    return "Path Strength: ⭐ Early signal"

def human_overlap_label(sim: float) -> str:
    if sim >= 0.55: return "Easy (high overlap)"
    if sim >= 0.40: return "Moderate"
    if sim >= 0.25: return "Stretch"
    return "Heavy lift"

def _palette_light():
    return dict(
        th_bg="#F9FAFB",
        td_hover="#F8FAFF",
        ink="#0f172a",
        border="#e5e7eb",
        tip_bg="#111827",
        tip_ink="#ffffff",
        ring="#cbd5e1",
        shadow="0 1px 10px rgba(16,24,40,.06)",
        zebra="#F3F6FD",
    )

def stepper_diagram(nodes: List[str], sims: Optional[List[float]] = None):
    """Vertical stepper (auto height, arrows always pointing DOWN without overlapping boxes)."""
    n = max(1, len(nodes))
    px_h = 360 + max(0, n - 2) * 160
    y_margin = 0.10
    gap = (1 - 2 * y_margin) / max(1, n - 1)

    # palette (light)
    node_colors = ["#2563EB", "#14B8A6", "#06B6D4", "#10B981", "#84CC16"]
    stroke = "rgba(0,0,0,0.16)"
    shadow = "rgba(2,6,23,0.18)"
    arrow = "rgba(148,163,184,0.65)"

    # Y positions from top → bottom
    ys = list(reversed([y_margin + i * gap for i in range(n)]))

    fam = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
    max_len = max(len(s) for s in nodes) if nodes else 10
    char_w = 0.012
    box_w = min(0.75, max(0.30, max_len * char_w))
    box_h = 0.18
    half_w, half_h = box_w / 2, box_h / 2
    x_mid = 0.5

    shapes, ann = [], []

    # Draw boxes + labels
    for i, (label, y) in enumerate(zip(nodes, ys)):
        c = node_colors[min(i, len(node_colors) - 1)]
        x0, x1 = x_mid - half_w, x_mid + half_w
        y0, y1 = y - half_h, y + half_h

        # shadow
        shapes.append(dict(
            type="rect", x0=x0, y0=y0 - 0.012, x1=x1, y1=y1 - 0.012,
            xref="x", yref="y", line=dict(color="rgba(0,0,0,0)"),
            fillcolor=shadow, layer="below", opacity=0.35
        ))
        # box
        shapes.append(dict(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            xref="x", yref="y", line=dict(color=stroke, width=1.2),
            fillcolor=c, layer="below"
        ))
        ann.append(dict(
            x=x_mid, y=y, xref="x", yref="y",
            text=f"<b>{html.escape(label)}</b>",
            showarrow=False, font=dict(color="white", size=16, family=fam),
            xanchor="center", yanchor="middle"
        ))

    # Draw connectors + arrowheads DOWN
    head_w = 0.014
    head_h = 0.020
    gap_offset = 0.015  # extra padding so lines don’t touch boxes

    for i in range(n - 1):
        y_start = ys[i] - half_h - gap_offset      # just below current box
        y_end   = ys[i + 1] + half_h + gap_offset  # just above next box

        shapes.append(dict(
            type="line",
            x0=x_mid, y0=y_start, x1=x_mid, y1=y_end,
            line=dict(color=arrow, width=10),
            layer="below"
        ))
        shapes.append(dict(
            type="path",
            path=f"M {x_mid - head_w} {y_end + head_h} "
                 f"L {x_mid} {y_end} "
                 f"L {x_mid + head_w} {y_end + head_h} Z",
            fillcolor=arrow, line=dict(color=arrow),
            layer="above"
        ))

    fig = go.Figure()
    fig.update_layout(
        height=px_h, margin=dict(l=10, r=10, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        shapes=shapes, annotations=ann
    )
    return fig

def render_skill_group(title: str, items: List[str], kind: str):
    st.markdown(f'<div class="section-sub">{html.escape(title)}</div>', unsafe_allow_html=True)
    if not items:
        st.markdown('<div class="skills-box">—</div>', unsafe_allow_html=True); return
    chips = " ".join([f'<span class="badge {kind}">{html.escape(s)}</span>' for s in items])
    st.markdown(f'<div class="skills-box">{chips}</div>', unsafe_allow_html=True)

def render_learning_links(unlock_skills: List[str], added: List[str]):
    pruned = [s for s in unlock_skills if s not in set(added)]
    links = get_learning_links(pruned, max_per_skill=2, max_total=10)
    if not links: return
    st.markdown("##### Learning resources")
    by_skill = {}
    for it in links:
        by_skill.setdefault(it["skill"], []).append(it)
    for skill, items in by_skill.items():
        st.markdown(f"- **{skill}**")
        for it in items:
            st.markdown(f'  - {it["kind"]} · [{it["label"]}]({it["url"]})')

def html_table(headers, rows, center_numeric: bool = False):
    """
    Beautiful table in an iframe; honors per-column alignment:
    - header['align'] can be 'left' (default), 'center', or 'right'
    - numeric columns align exactly under their headers
    """
    C = _palette_light()
    style = f"""
    <style>
      .tbl-wrap {{
        position:relative; border-radius:14px; box-shadow:{C['shadow']};
        border:1px solid {C['border']}; overflow: visible;
      }}
      .tbl{{width:100%; border-collapse:separate; border-spacing:0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;}}
      .tbl th, .tbl td{{ padding:14px 16px; border-bottom:1px solid {C['border']}; font-size:15px; color:{C['ink']}; }}
      .tbl th{{ background:{C['th_bg']}; text-align:left; font-weight:700; font-size:16px; }}
      .tbl tbody tr:nth-child(even) td{{ background:{C['zebra']}; }}
      .tbl tr:hover td{{ background:{C['td_hover']} !important; }}

      /* alignment helpers */
      .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
      .num.center {{ text-align:center; }}
      .num.left {{ text-align:left; }}

      .col-index{{ width:44px; text-align:center; color:#64748b; font-weight:600; }}
      .th-help{{ position:relative; white-space:nowrap; }}
      .qm{{ display:inline-flex; align-items:center; justify-content:center;
           width:20px; height:20px; margin-left:8px; border-radius:50%;
           border:1px solid {C['ring']}; background:transparent; color:#9ca3af; font-weight:800;
           line-height:1; cursor:pointer; font-size:12px; }}
      .th-help .tip{{
        position:absolute; left:0; top:calc(100% + 10px);
        background:{C['tip_bg']}; color:{C['tip_ink']};
        border-radius:10px; padding:10px 12px;
        font-size:12px; line-height:1.35; max-width:520px;
        box-shadow:0 10px 24px rgba(0,0,0,.22); display:none; z-index:9999; white-space:normal;
      }}
      .th-help:hover .tip, .th-help .qm:focus + .tip{{ display:block; }}
    </style>
    """

    def cls_for(align: str) -> str:
        # map align to class used for both th and td
        if align == "center":
            return "num center"
        if align == "right":
            return "num"
        if align == "left":
            return "num left"
        return ""  # default left

    # THEAD
    thead = []
    for h in headers:
        label = html.escape(h["label"])
        align = h.get("align", "left")
        cls = cls_for(align) if align in ("left", "center", "right") else ""
        if h.get("help"):
            tip = html.escape(h["help"])
            thead.append(
                f'<th class="th-help {cls}">{label}'
                f'  <button class="qm" aria-label="Help">?</button>'
                f'  <div class="tip">{tip}</div>'
                f'</th>'
            )
        else:
            thead.append(f'<th class="{cls}">{label}</th>')

    # TBODY
    body_rows = []
    for r in rows:
        tds = []
        for (val, h) in zip(r, headers):
            align = h.get("align", "left")
            cls = cls_for(align) if align in ("left", "center", "right") else ""
            tds.append(f'<td class="{cls}">{html.escape(str(val))}</td>')
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    return (
        style
        + "<div class='tbl-wrap'><table class='tbl'><thead><tr>"
        + "".join(thead)
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )

# ---------------- Session Defaults ----------------
if "current_prefill" not in st.session_state: st.session_state["current_prefill"] = None
if "resume_skills" not in st.session_state:  st.session_state["resume_skills"] = []
if "sandbox_skills_store" not in st.session_state: st.session_state["sandbox_skills_store"] = []

# ---------------- Résumé (optional) ----------------
with st.expander("Expand to add your Résumé (optional)", expanded=False):
    c1, c2 = st.columns([0.9, 1.3])  # more space for the table
    with c1:
        up = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
        pasted = st.text_area("…or paste plain text", height=130, placeholder="Paste your résumé text here")
        resume_text = parse_resume(up, pasted)
        if up: st.caption(f"Parsed **{up.name}**")

    with c2:
        if resume_text:
            detected = extract_resume_skills(resume_text)
            st.session_state["resume_skills"] = detected[:]
            st.markdown(f"#### Skills detected in résumé · {len(detected)}")
            chips = " ".join([f'<span class="badge detect">{html.escape(s)}</span>' for s in detected])
            st.markdown(f'<div class="skills-box" style="text-align:left;">{chips}</div>', unsafe_allow_html=True)

            ranked = score_roles_by_resume(resume_text, topn=8)
            if ranked:
                headers = [
                    {"label":"Suggested role"},
                    {"label":"Overall likelihood", "help":"How promising this role is for you overall, compared to other roles. Example: 7.9% ≈ ~8 out of 100 options for your profile.", "align":"center"},
                    {"label":"Résumé match (0–1)", "help":"How much of this role’s skills appear in your résumé (0–1). Higher is closer.", "align":"center"},
                    {"label":"Matched skills (#)", "align":"center"}
                ]
                rows = [(r["role"], f"{r['prob']*100:.1f}%", f"{r['score']:.2f}", len(r["matched_skills"])) for r in ranked]
                components.html(html_table(headers, rows, center_numeric=False), height=min(560, 140 + 44*len(rows)), scrolling=True)

                options = [r["role"] for r in ranked]
                chosen = st.selectbox("Use this as your current role:", options, index=0)
                if st.button("Set current role"):
                    st.session_state["current_prefill"] = chosen
                    st.success(f"Current role set to: {chosen}")
            else:
                st.info("Couldn’t find clear role matches in the résumé. You can still pick roles manually below.")
        else:
            st.caption("Upload a résumé or paste text to get automatic suggestions.")

st.markdown('<div class="hard-divider"></div>', unsafe_allow_html=True)

# ---------------- Role selectors ----------------
prefill_role = st.session_state.get("current_prefill")
options_current = ["—"] + ROLES
cur_index = options_current.index(prefill_role) if prefill_role in options_current else 0

col1, col2 = st.columns(2)
with col1: cur = st.selectbox("Current role", options=options_current, index=cur_index)
with col2: tar = st.selectbox("Target role", options=["—"] + ROLES)

st.markdown('<div class="hard-divider"></div>', unsafe_allow_html=True)

# ---------------- Skills you have ----------------
st.subheader("My current skills")
resume_skills_available = st.session_state.get("resume_skills", [])
added_skills = st.multiselect(
    "Add or remove skills",
    options=SKILLS,
    default=st.session_state["sandbox_skills_store"]
)
if resume_skills_available:
    if st.button(f"Add skills from résumé ({len(resume_skills_available)} found)", key="add_resume_skills"):
        merged = sorted(set(st.session_state["sandbox_skills_store"]) | set(resume_skills_available))
        st.session_state["sandbox_skills_store"] = merged
        st.success(f"Added {len(resume_skills_available)} skills from résumé.")
        st.rerun()
if added_skills != st.session_state["sandbox_skills_store"]:
    st.session_state["sandbox_skills_store"] = added_skills[:]

st.markdown('<div class="hard-divider"></div>', unsafe_allow_html=True)

# ---------------- Recommended next roles ----------------
if cur and cur != "—":
    st.subheader(f"Top next-step suggestions from {cur}")
    if added_skills:
        df_next = neighbors_with_added_skills(cur, added_skills, topn=10)
    else:
        try:
            df_next = neighbors(cur, topn=10, target=(tar if tar != "—" else None))
        except TypeError:
            df_next = neighbors(cur, topn=10)

    if df_next is None or df_next.empty:
        st.info("No strong neighbors at the current threshold.")
    else:
        show = df_next.copy().sort_values("score", ascending=False)
        rows = [(row["dst"], f"{row['score']:.2f}", f"{row['sim']:.2f}") for _, row in show.iterrows()]
        headers = [
            {"label":"Suggested role"},
            {"label":"Opportunity score", "help":"Blends skill match with current market demand for the role. Higher is better.", "align":"center"},
            {"label":"Skill match (0–1)", "help":"How closely your current skills match that role (0–1). Higher is a closer match.", "align":"center"},
        ]
        components.html(html_table(headers, rows, center_numeric=True), height=min(520, 130 + 44*len(rows)), scrolling=True)

st.markdown('<div class="hard-divider"></div>', unsafe_allow_html=True)

# ---------------- Career paths ----------------
if cur and cur != "—" and tar and tar != "—":
    st.subheader(f"Paths from {cur} → {tar}")

    def dynamic_paths_with_added(start, goal, added, max_paths=3):
        if start == goal: return [(1.0, [start], [])]
        A = augment_current_with_skills(start, added)
        direct = weighted_jaccard(A, _ROLE2SK.get(goal, {}))
        candidates = []
        for r, sk in _ROLE2SK.items():
            if r in (start, goal): continue
            s = weighted_jaccard(A, sk)
            if s > 0.05: candidates.append((r, s))
        candidates.sort(key=lambda x: x[1], reverse=True)
        paths=[]
        if direct > 0.10: paths.append((direct, [start, goal], [direct]))
        for inter, s1 in candidates[:10]:
            s2 = role_similarity(inter, goal)
            if s2 > 0.05:
                paths.append(((s1+s2)/2, [start, inter, goal], [s1, s2]))  # keep order start->inter->goal
        paths.sort(key=lambda x:x[0], reverse=True)
        return paths[:max_paths]

    if added_skills:
        paths = dynamic_paths_with_added(cur, tar, added_skills, max_paths=3)
        if paths: st.success("✨ Paths updated using your added skills.")
    else:
        paths = get_paths(cur, tar, static_depth=3, static_beam=6, dyn_depth=4, dyn_beam=10)

    def order_path(nodes, sims, start, goal):
        """
        Ensure the diagram (and the skill sections) always run start → ... → goal.
        Re-slice sims to match the chosen direction.
        """
        if not nodes or start not in nodes or goal not in nodes or len(nodes) < 2:
            return nodes, sims
        i_cur, i_tar = nodes.index(start), nodes.index(goal)
        if i_cur <= i_tar:
            # forward slice
            new_nodes = nodes[i_cur:i_tar+1]
            new_sims  = sims[i_cur:i_tar] if sims else []
        else:
            # reverse slice
            new_nodes = list(reversed(nodes[i_tar:i_cur+1]))
            new_sims  = list(reversed(sims[i_tar:i_cur])) if sims else []
        return new_nodes, new_sims

    if not paths:
        st.warning("No good path found. Try a nearby target, or add one intermediate step.")
    else:
        for i, (avg_score, nodes, sims) in enumerate(paths, start=1):
            nodes, sims = order_path(nodes, sims, cur, tar)

            col_left, col_right = st.columns([1.5, 2.0])
            with col_left:
                st.markdown(f"**Path {i} · {strength_label(avg_score)}**")
                st.caption(" → ".join(nodes))
                steps = len(nodes) - 1
                st.markdown(
                    f'<div class="kpi"><b>Steps:</b> {steps}&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'<b>Overall Fit:</b> {int(np.clip(avg_score,0,1)*100)}%</div>',
                    unsafe_allow_html=True
                )
                st.plotly_chart(
                    stepper_diagram(nodes, sims),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

            with col_right:
                st.markdown("#### Why this path works (skills view)")
                for j, (a, b, s) in enumerate(zip(nodes[:-1], nodes[1:], sims)):
                    st.markdown(
                        f"**{html.escape(a)} → {html.escape(b)}**  "
                        f"<span style='color:var(--muted);font-family:ui-monospace,Menlo,Consolas'>· transition strength {s:.2f}</span>",
                        unsafe_allow_html=True
                    )

                    # ---------- UPDATED LOGIC: always surface newly added skills in "Transferable strengths" ----------
                    if added_skills:
                        # Treat added skills as owned on every hop, and make sure they appear in "Transferable strengths"
                        A = augment_current_with_skills(a, added_skills)
                        B = role_skills(b)

                        have        = set(A)
                        target_set  = set(B)
                        added_set   = set(added_skills)

                        # Added skills that are relevant to this step appear first
                        added_carry = sorted(added_set & target_set, key=lambda x: B.get(x, 0), reverse=True)
                        # Other overlaps you already have
                        other_carry = sorted((have & target_set) - set(added_carry), key=lambda x: B.get(x, 0), reverse=True)

                        carry_full  = added_carry + other_carry
                        carry       = carry_full[:8]

                        # Skills still missing for the next role (exclude what you have, including newly added)
                        need        = sorted(target_set - have, key=lambda x: B.get(x, 0), reverse=True)[:8]
                    else:
                        carry, need = explain_step(a, b, top_k=8)
                    # -----------------------------------------------------------------------------------------------

                    st.markdown(
                        f"<div class='kpi' style='margin:.25rem 0 .5rem 0;'>"
                        f"Skill overlap: <b>{int(np.clip(s,0,1)*100)}%</b>&nbsp;&nbsp;·&nbsp;&nbsp;"
                        f"Gap to close: <b>{len(need)} new skill{'s' if len(need)!=1 else ''}</b>&nbsp;&nbsp;·&nbsp;&nbsp;"
                        f"{human_overlap_label(s)}</div>",
                        unsafe_allow_html=True
                    )
                    render_skill_group("Transferable strengths", carry, "strength")
                    render_skill_group("Skills to unlock next", need, "unlock")
                    render_learning_links(need, added_skills)

                    if j < len(sims)-1:
                        st.markdown("<hr style='border:none;height:1px;background:var(--border);margin:12px 0;'>", unsafe_allow_html=True)

            st.markdown('<div class="hard-divider"></div>', unsafe_allow_html=True)

        if added_skills:
            with st.expander("Impact of added skills"):
                st.markdown(f"**Added skills:** {', '.join(added_skills)}")
                st.markdown("• Path strengths recalculated with your enhanced profile")
                st.markdown("• Transferable strengths updated for every hop")
                st.markdown("• Recommendations reordered by compatibility")
