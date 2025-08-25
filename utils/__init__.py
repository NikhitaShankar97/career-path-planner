import pandas as pd
import numpy as np
import re
import io
from typing import List, Tuple, Dict, Optional
from PyPDF2 import PdfReader
from docx import Document

# -------------------- Data --------------------
RS  = pd.read_csv("data/role_skills_specific.csv")   # role, skill, weight
EDG = pd.read_csv("data/transitions.csv")            # src, dst, sim
try:
    DEM = pd.read_csv("data/demand.csv").set_index("role")["demand_pct"].to_dict()
except Exception:
    DEM = {}

ROLES  = sorted(RS["role"].unique().tolist())
SKILLS = sorted(RS["skill"].unique().tolist())

# Pre-compute role -> {skill:weight}
_ROLE2SK: Dict[str, Dict[str, int]] = {
    r: {row.skill: int(row.weight) for row in g.itertuples(index=False)}
    for r, g in RS.groupby("role")
}

# -------------------- Core helpers --------------------
def role_skills(role: str) -> Dict[str, int]:
    g = RS[RS["role"] == role][["skill", "weight"]]
    return dict(zip(g["skill"], g["weight"]))

def weighted_jaccard(a: Dict[str,int], b: Dict[str,int]) -> float:
    keys = set(a) | set(b)
    num  = sum(min(a.get(k,0), b.get(k,0)) for k in keys)
    den  = sum(max(a.get(k,0), b.get(k,0)) for k in keys)
    return (num/den) if den else 0.0

def role_similarity(role_a: str, role_b: str) -> float:
    return weighted_jaccard(_ROLE2SK.get(role_a, {}), _ROLE2SK.get(role_b, {}))

def edge_score(src: str, dst: str, sim: float, lam: float = 0.3) -> float:
    return float(sim) * (1 + lam * float(DEM.get(dst, 0.0)))

# -------------------- Static graph APIs --------------------
def neighbors(role: str, topn: int = 10, lam: float = 0.3, target: Optional[str] = None):
    df = EDG[EDG["src"] == role].copy()
    if not df.empty:
        df["score"] = df.apply(lambda r: edge_score(r["src"], r["dst"], r["sim"], lam), axis=1)
        return df.sort_values("score", ascending=False).head(topn)[["dst","score","sim"]]
    dyn = dynamic_neighbors(role, target=target, k=topn, min_sim=0.03, bias=0.30)
    if not dyn:
        return pd.DataFrame(columns=["dst","score","sim"])
    return pd.DataFrame([(d, s, s) for d, s in dyn], columns=["dst","score","sim"])

def best_paths(start: str, goal: str, max_depth: int = 3, beam: int = 6, lam: float = 0.3):
    if start == goal:
        return [(1.0, [start], [])]
    nxt: Dict[str, List[Tuple[str,float]]] = {}
    for _, r in EDG.iterrows():
        nxt.setdefault(r["src"], []).append((r["dst"], float(r["sim"])))
    frontier: List[Tuple[float, List[str], List[float]]] = [(0.0, [start], [])]
    best: List[Tuple[float, List[str], List[float]]] = []
    for _ in range(max_depth):
        new: List[Tuple[float, List[str], List[float]]] = []
        for _, nodes, sims in frontier:
            cur = nodes[-1]
            for dst, sim in sorted(nxt.get(cur, []), key=lambda x: x[1], reverse=True)[:beam]:
                if dst in nodes:
                    continue
                s  = edge_score(cur, dst, sim, lam)
                ns = nodes + [dst]
                ss = sims + [s]
                avg = sum(ss) / len(ss)
                if dst == goal:
                    best.append((avg, ns, ss))
                else:
                    new.append((avg, ns, ss))
        new.sort(key=lambda x: x[0], reverse=True)
        frontier = new[:beam]
    best.sort(key=lambda x: x[0], reverse=True)
    return best[:3]

# -------------------- Explanations & what-if --------------------
def explain_step(a: str, b: str, top_k: int = 6):
    A = role_skills(a); B = role_skills(b)
    carry = sorted(set(A) & set(B), key=lambda s: B[s], reverse=True)[:top_k]
    add   = sorted(set(B) - set(A), key=lambda s: B[s], reverse=True)[:top_k]
    return carry, add

def augment_current_with_skills(cur: str, extra: List[str], default_weight: int = 4) -> Dict[str,int]:
    base = role_skills(cur).copy()
    for s in extra:
        base[s] = max(base.get(s, 0), default_weight)
    return base

def neighbors_with_added_skills(cur: str, added: List[str], topn: int = 10, lam: float = 0.3):
    df_static = EDG[EDG["src"] == cur]
    if df_static.empty:
        A = augment_current_with_skills(cur, added)
        rows = []
        for other in ROLES:
            if other == cur:
                continue
            sim_aug = weighted_jaccard(A, _ROLE2SK.get(other, {}))
            rows.append((other, sim_aug, sim_aug))
        df = pd.DataFrame(rows, columns=["dst","score","sim"])
        return df.sort_values("score", ascending=False).head(topn)
    A = augment_current_with_skills(cur, added)
    rows = []
    for _, r in df_static.iterrows():
        dst = r["dst"]
        sim_aug = weighted_jaccard(A, role_skills(dst))
        rows.append((dst, sim_aug * (1 + lam * float(DEM.get(dst, 0.0))), sim_aug))
    df = pd.DataFrame(rows, columns=["dst","score","sim"])
    return df.sort_values("score", ascending=False).head(topn)

# -------------------- Universal dynamic graph --------------------
def dynamic_neighbors(role: str,
                      target: Optional[str] = None,
                      k: int = 25,
                      min_sim: float = 0.05,
                      bias: float = 0.35) -> List[Tuple[str, float]]:
    base = _ROLE2SK.get(role, {})
    if not base:
        return []
    sims: List[Tuple[str, float]] = []
    for other, sk in _ROLE2SK.items():
        if other == role:
            continue
        s = weighted_jaccard(base, sk)
        if target is not None:
            s = s + bias * role_similarity(other, target)
        if s >= min_sim:
            sims.append((other, float(s)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

def robust_best_paths(start: str, goal: str, max_depth: int = 4, beam: int = 8,
                      k_per_step: int = 25, min_sim: float = 0.05, return_top: int = 3):
    if start == goal:
        return [(1.0, [start], [])]
    def H(r: str) -> float:
        return role_similarity(r, goal)
    beam_states: List[Tuple[float, float, List[str], List[float]]] = [(H(start), 0.0, [start], [])]
    solutions: List[Tuple[float, List[str], List[float]]] = []
    for _ in range(max_depth):
        new_states: List[Tuple[float, float, List[str], List[float]]] = []
        seen = set()
        for _, avg_s, path, sims in beam_states:
            cur = path[-1]
            for nbr, sc in dynamic_neighbors(cur, target=goal, k=k_per_step, min_sim=min_sim, bias=0.35):
                if nbr in path:
                    continue
                sims2 = sims + [sc]
                path2 = path + [nbr]
                if nbr == goal:
                    solutions.append((float(np.mean(sims2)), path2, sims2))
                    continue
                avg2 = float(np.mean(sims2))
                pr   = avg2 + 0.5 * H(nbr)
                key  = (nbr, len(path2))
                if key in seen: continue
                seen.add(key)
                new_states.append((pr, avg2, path2, sims2))
        if solutions: break
        if not new_states: min_sim = max(0.01, min_sim * 0.7)
        new_states.sort(key=lambda x: x[0], reverse=True)
        beam_states = new_states[:beam]
    if solutions:
        solutions.sort(key=lambda x: x[0], reverse=True)
        return solutions[:return_top]
    # Greedy backup
    cur = start; path = [cur]; sims: List[float] = []
    for _ in range(max_depth):
        neigh = dynamic_neighbors(cur, target=goal, k=k_per_step, min_sim=0.01, bias=0.45)
        neigh = [n for n in neigh if n[0] not in path]
        if not neigh: break
        nxt, sc = neigh[0]
        path.append(nxt); sims.append(sc); cur = nxt
        if cur == goal: break
    if path[-1] != goal:
        sc = max(0.01, role_similarity(path[-1], goal) * 0.8)
        path.append(goal); sims.append(sc)
    return [(float(np.mean(sims)) if sims else 0.0, path, sims)]

# -------------------- Unified path API --------------------
def get_paths(start: str, goal: str,
              static_depth: int = 3, static_beam: int = 6,
              dyn_depth: int = 4, dyn_beam: int = 10) -> List[Tuple[float, List[str], List[float]]]:
    paths = best_paths(start, goal, max_depth=static_depth, beam=static_beam)
    if paths: return paths
    return robust_best_paths(start, goal, max_depth=dyn_depth, beam=dyn_beam, k_per_step=30, min_sim=0.05)

# -------------------- Résumé parsing --------------------
def _read_pdf_bytes(b: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""

def _read_docx_bytes(b: bytes) -> str:
    try:
        doc = Document(io.BytesIO(b))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def parse_resume(uploaded_file, pasted_text: str = "") -> str:
    if uploaded_file is not None:
        name = (uploaded_file.name or "").lower()
        data = uploaded_file.read()
        if name.endswith(".pdf"):
            txt = _read_pdf_bytes(data);  return txt.strip() if txt.strip() else ""
        elif name.endswith(".docx"):
            txt = _read_docx_bytes(data); return txt.strip() if txt.strip() else ""
        elif name.endswith(".txt"):
            try: return data.decode("utf-8", errors="ignore").strip()
            except Exception: pass
    return (pasted_text or "").strip()

# ---------- Skill extraction ----------
_SKILL_PAT = { s: re.compile(rf"\b{re.escape(s.lower())}\b") for s in SKILLS }
_ALIASES = { "sklearn":"Scikit-learn","tf":"TensorFlow","pytorch":"PyTorch","mlops":"ML Ops","ml ops":"ML Ops","bi":"Power BI",
             "postgres":"PostgreSQL","gcp":"GCP","aws":"AWS","ms sql":"SQL" }

def extract_resume_skills(text: str) -> list[str]:
    if not text: return []
    t = text.lower(); hits = set()
    for s, rx in _SKILL_PAT.items():
        if s.lower() in t and rx.search(t): hits.add(s)
    for alias, canon in _ALIASES.items():
        if alias in t and canon in SKILLS: hits.add(canon)
    return sorted(hits)

# ---------- Role suggestion ----------
_ROLE_TOTAL = RS.groupby("role")["weight"].sum().to_dict()

def score_roles_by_resume(text: str, topn: int = 8, temperature: float = 0.7):
    import numpy as _np
    found = set(extract_resume_skills(text))
    if not found: return []
    rows = []
    for role, g in RS.groupby("role"):
        skills = {r.skill: int(r.weight) for r in g.itertuples(index=False)}
        overlap = [s for s in skills if s in found]
        if not overlap: continue
        numer = sum(skills[s] for s in overlap)
        denom = max(1, int(_ROLE_TOTAL.get(role, 1)))
        score = numer / denom
        rows.append((role, score, overlap))
    if not rows: return []
    scores = _np.array([r[1] for r in rows], dtype=float)
    logits = scores / max(1e-6, float(temperature))
    e = _np.exp(logits - logits.max()); probs = e / e.sum()
    ranked = sorted(
        [{"role": role, "prob": float(p), "score": float(sc), "matched_skills": overlaps}
         for (role, sc, overlaps), p in zip(rows, probs)],
        key=lambda x: x["prob"], reverse=True
    )[:topn]
    return ranked

__all__ = ["ROLES","SKILLS","neighbors","best_paths","explain_step","augment_current_with_skills",
           "neighbors_with_added_skills","get_paths","role_similarity","weighted_jaccard","_ROLE2SK",
           "role_skills","parse_resume","extract_resume_skills","score_roles_by_resume"]
# utils/__init__.py

# Minimal learning links (extend freely)
_SKILL_LINKS = {
    "SQL": [
        {"kind": "Tutorial", "label": "Mode SQL School", "url": "https://mode.com/sql-tutorial/"},
        {"kind": "Course", "label": "Khan Academy SQL", "url": "https://www.khanacademy.org/computing/computer-programming/sql"}
    ],
    "Python": [
        {"kind": "Docs", "label": "Official Python Tutorial", "url": "https://docs.python.org/3/tutorial/"},
        {"kind": "Guide", "label": "Python.org Getting Started", "url": "https://www.python.org/about/gettingstarted/"}
    ],
    "Machine Learning": [
        {"kind": "Docs", "label": "scikit-learn Tutorial", "url": "https://scikit-learn.org/stable/tutorial/index.html"},
        {"kind": "Course", "label": "Coursera: Andrew Ng ML", "url": "https://www.coursera.org/learn/machine-learning"}
    ],
    "Pandas": [
        {"kind": "Docs", "label": "Pandas User Guide", "url": "https://pandas.pydata.org/docs/user_guide/index.html"}
    ],
    "NumPy": [
        {"kind": "Docs", "label": "NumPy Learn", "url": "https://numpy.org/learn/"}
    ],
    "Scikit-learn": [
        {"kind": "Docs", "label": "scikit-learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html"}
    ],
    "TensorFlow": [
        {"kind": "Docs", "label": "TensorFlow Guide", "url": "https://www.tensorflow.org/guide"}
    ],
    "Experiment Design": [
        {"kind": "Intro", "label": "A/B Testing basics", "url": "https://www.optimizely.com/optimization-glossary/ab-testing/"}
    ]
}

def get_learning_links(skills, max_per_skill=2, max_total=12):
    """
    Return up to max_total learning links across given skills.
    Each item will have: {"skill": str, "label": str, "url": str, "kind": str}
    """
    out = []
    for s in skills:
        if s in _SKILL_LINKS:
            for item in _SKILL_LINKS[s][:max_per_skill]:
                out.append({
                    "skill": s,
                    "label": item["label"],
                    "url": item["url"],
                    "kind": item["kind"],
                })
                if len(out) >= max_total:
                    return out
    return out
