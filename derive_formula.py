"""
derive_formula.py — ask DeepSeek-R1 to COMPRESS the Constellax pipeline (A->B) into the
single conserved operation we want to distill into a small model, then draft, derive,
prove, fit-test, and adjudicate candidate mathematical criteria for it.

ISOLATED + SAFE. Touches NO production code. Reads only OPENROUTER_API_KEY from the
reasoningEngine .env, calls OpenRouter directly via httpx, writes only into ./out/.
Save-as-you-go: each R1 stage is flushed to disk the moment it returns, so a late
timeout never loses earlier work.

This is NOT the descriptive formalization (that already exists in
~/Desktop/pipeline-math/pipeline_math.md). This derives a TRAINABLE CRITERION with
explicit provenance back to pipeline stages, proofs, a fit-test, and a trainability
check for the IBM Granite 3B-dense / 7B-hybrid models.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

REASONING_ENV = Path.home() / "Desktop" / "reasoningEngine" / ".env"
load_dotenv(str(REASONING_ENV))
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("OPENROUTER_API_KEY not found in " + str(REASONING_ENV))

MODEL = "deepseek/deepseek-r1"
URL = "https://openrouter.ai/api/v1/chat/completions"
OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)
MASTER = OUTDIR / "derivation.md"
TIMEOUT = 300.0
RETRIES = 3

# --- GROUND TRUTH: the complete pipeline A->B -------------------------------------
# Copied verbatim from the vetted SEGMENTS in
# ~/Desktop/pipeline-math/formalize_pipeline.py so R1 reasons about the real pipeline.
PIPELINE = """\
The pipeline is an autonomous multi-agent reasoning loop. Its foundational principle is
DIVERGENT THINKING: see a problem from many independent angles and DELIBERATELY DO NOT
converge them into one answer; surface multiple distinct threads (each with its own logic
and its own projected consequences) and let the human pick which thread(s) to pull.

SEGMENT 0 — INPUT / CUSHION + LAUNDERING. The intake is a 'cushion' = an ordered tuple
(problem, context, vision, hunches, question). The QUESTION is deliberately withheld from
the divergent stage (a 'chaos law': the generators must not see the goal, to stay
divergent). The question is parsed into k sub-questions/angles. Each sub-question is passed
through a 'launderer' (a leak-gate translator) that rewrites it into a goal-free 'lead' —
it strips interrogative form, goal-verbs, and any n-gram overlap with the question, failing
closed (drop) if it cannot. Output: an anchor A = (problem, context, vision, hunches)
shared by all agents, plus k laundered leads L_1..L_k, one per agent.

SEGMENT 1 — WANDERING (divergent generation). n agents run in parallel (n = number of
sub-questions). Agent i is seeded with a per-agent cushion = anchor A with its problem
field augmented by lead L_i. Each agent is GOAL-BLIND (never sees the question). An agent
performs repeated 'digs': it searches across many distant knowledge domains for a source
whose STRUCTURE is analogous to the anchor's structure (not its surface topic), and judges
the structural match with an LLM. Each accepted match becomes a 'card' carrying:
source_shape, the structural bridge to the anchor, and a match strength. Agents post to a
shared noticeboard. A 'governor' reads the noticeboard, builds a skeleton of the emerging
structure, and emits a control signal in {HOLD, CLOSE}: CLOSE halts the wave when the
structure has converged. Output: a growing set of cards C_t and a skeleton with 'gaps'.

SEGMENT 2 — MIDDLE (audit, checkpoint, steer, iterate). Once per cycle on accumulated
cards. (a) HALO auditor: finds blind spots the cards miss. (b) COVERAGE checkpoint: the
ONLY goal-aware judge in the loop; scores D_t in [0,1] = fraction of the k question-angles
the cards now cover, and lists which angles remain open. (c) SHEPHERD: a trajectory sensor
returning {on_track, circling, drifting} + a refocus nudge; advisory, never halts.
(d) DISPATCHER: fuses signals in priority order — skeleton gaps, high-severity halo blind
spots, open coverage angles, shepherd refocus — into fresh leads for the next cycle, each
laundered again. Loop wander->middle repeats until D_t is high and settled, or a hard cycle
cap (<=4) is hit, or no fresh leads remain. Output per cycle: D_t, open angles, next leads.

SEGMENT 3 — BLENDER (convergent synthesis). Runs ONCE after the loop, on all cards.
Goal-AWARE. Two seats (creative model 'Opus', critic model 'R1') run a 4-round protocol:
draft candidate fusions, critique each other's, emit final fusions with an agreement
status, give disputed fusions per-seat angles. A 'fusion' combines >=2 cards into a single
proposal: a thesis, a mechanism, and a claim about how it advances the cushion, with
citations to source cards. CRITICAL: the blender preserves DISTINCT theses — it does NOT
merge competing fusions into one blended answer or hedge between them. Cost-capped.
Coverage is re-scored on the BLENDS = the true D_t. Output: a set of fusions F.

SEGMENT 4 — R1 FORMALIZER. Junior to the blender (reads fusions, never rewrites). For each
fusion: is it formalizable (yes/partial/no), and if so ground it into a formal core
(objects, symbols, candidate equation), citations mapping each symbol to a source card, an
honest list of what stays qualitative, and a predict/confirm/falsify test.

THE UNIFIED MAP (from the earlier descriptive formalization): the pipeline is a composition
Phi: cushion -> {formalized proposals}, with a per-cycle loop wander->middle (halt when
coverage high-and-settled, or cycle cap <=4, or no leads), then the once-after blender and
formalizer. The goal-blind/goal-aware split is the spine: laundering hides the question
from the wander; only coverage and the blender may see it. The conserved theme end-to-end
is EXPANSION (many independent structurally-grounded threads) held OPEN, never contracted
to a single argmax answer.
"""

SHARED_CONTEXT = """\
PROJECT — Constellax: an autonomous multi-agent reasoning pipeline (point A to point B,
described below). Foundational principle: DIVERGENT THINKING — many independent angles,
deliberately NOT converged; the human picks the thread.

GOAL OF THIS WORK — we are DISTILLING the pipeline's single conserved operation into a
SMALL language model via supervised fine-tuning, as a research demonstration + portfolio
thesis. Models: IBM Granite 4.0 — (1) granite-4.0-micro, 3B pure DENSE transformer, and
(2) granite-4.0-h-tiny, 7B HYBRID (Mamba-2 + MoE). They are small, so we can train them on
exactly ONE narrow, well-defined STRUCTURAL behavior — not general wisdom. (We measure the
structural property, divergence; we do NOT claim the reasoning is wise on a model this
small.)

THE ONE TRAINING TASK (the only thing the model must learn): given a problem, do not race
to a single fast answer. Reason from MULTIPLE independent angles; for each angle, project
its CONSEQUENCES (every path has a consequence); emit MULTIPLE non-converged THREADS; never
blend or collapse them; let the user choose the direction. Wide base, narrowed by the
USER's own iteration — a pyramid the user descends, not the model.

WHY A FORMULA, AND WHY STRICT — we want the pipeline's conserved operation as a precise
mathematical criterion. Strictness is INTENTIONAL and desirable: a strict criterion
collapses the training target onto a narrow manifold, which is exactly what a small model
needs to learn one thing reliably, and what ELIMINATES drift (a loose 'reason well' target
is a wide region the model wanders in; a sharp criterion is not). The criterion doubles as
(a) the training-example spec / reward / corpus-acceptance filter, and (b) the methodology
section of our paper — with full PROVENANCE: extracted from a pipeline we actually run, not
invented abstractly.

KNOWN DESIGN CONSTRAINT (respect it or explicitly refute it): a single-term criterion that
only MAXIMIZES divergence between threads is GAMEABLE — a model can satisfy it by emitting
maximally-distant but UNGROUNDED nonsense (a different failure: divergence-into-noise). So
the criterion must hold a TENSION: maximize inter-thread divergence SUBJECT TO each thread
staying grounded in the problem AND carrying a valid projected consequence. Hold both
poles; collapse neither.

THE COMPLETE PIPELINE (point A -> point B):
""" + PIPELINE


def stage_prompt(body: str) -> str:
    return SHARED_CONTEXT + "\n\nYOUR TASK NOW:\n" + body


P_DECIPHER = stage_prompt(
    "Strip away the agent machinery and identify the SINGLE conserved operation the whole "
    "pipeline exists to perform — the irreducible invariant we want to distill into the "
    "model's weights. State it precisely (derive it yourself; do not just repeat my prose). "
    "Then propose THREE genuinely DISTINCT candidate mathematical formula-families for this "
    "operation as a TRAINABLE objective/criterion, each from a DIFFERENT branch of "
    "mathematics (e.g. information-theoretic; geometric / determinantal-point-process; "
    "game- or decision-theoretic; dynamical-systems / variational). For each candidate give "
    "ONLY: (i) a short name, (ii) the core functional form in 1-2 lines of LaTeX, (iii) a "
    "one-sentence intuition, (iv) which pipeline stage(s) most support it. Do NOT expand "
    "them yet. Make sure the three are REAL alternatives, not one idea renamed. Each must "
    "be able to carry the divergence<->grounding<->consequence tension."
)


def p_candidate(i: int, decipher_text: str) -> str:
    return stage_prompt(
        f"From your decipher step (reproduced below), fully develop CANDIDATE #{i} only. "
        "Produce its COMPLETE treatment:\n"
        "(1) FORMAL SPEC — objects, spaces, and the exact objective/criterion in LaTeX; "
        "define every symbol; make the divergence / grounding / consequence terms explicit "
        "and show how the tension is encoded (constraint, Lagrangian, product, etc.).\n"
        "(2) GENESIS / PROVENANCE — derive it LINE BY LINE from the real pipeline: map each "
        "term and symbol to the specific stage it is extracted from (laundering, wander, "
        "governor CLOSE, coverage D_t, shepherd, blender distinct-thesis preservation, "
        "formalizer). The origin must be explicit and auditable — no abstract invention.\n"
        "(3) PROOFS — state and PROVE what is provable: e.g. the criterion is maximized iff "
        "threads are mutually distinct AND grounded; pure-divergence degenerates without the "
        "grounding term; the non-convergence (no-argmax) invariant is preserved; any bounds, "
        "monotonicity, or degeneracy results. Be rigorous; mark clearly proven vs asserted "
        "vs heuristic.\n"
        "(4) FIT-TEST against the REAL pipeline — does the criterion actually describe what "
        "the pipeline does end to end (wander expands, blender preserves distinct theses, "
        "nothing is force-converged)? Where does it match, where does it fail to capture the "
        "real behavior? Be honest about the gap.\n"
        "(5) CONTRIBUTION + TRAINABILITY — used as an SFT target / reward / corpus filter for "
        "a 3B dense and a 7B hybrid model, does it actually push the model toward the one "
        "task? Is it OPERATIONAL (differentiable, or usable as a reward, or as a "
        "corpus-acceptance filter at data-generation time)? Name its concrete failure mode "
        "and how you would DETECT it.\n"
        "Use LaTeX. Rigorous and honest — this goes into a paper with its provenance "
        "attached. End with a compact 6-line SUMMARY CARD (name; form; strongest provenance; "
        "strongest proof; biggest fit-gap; trainability verdict).\n\n"
        "=== DECIPHER STEP (for reference) ===\n" + decipher_text
    )


def p_adjudicate(cards: str) -> str:
    return stage_prompt(
        "Here are the three fully-developed candidate criteria (their summary cards and key "
        "content below). Compare them across FOUR axes: provenance strength (how cleanly "
        "each traces to the real pipeline), proof strength, fit to actual pipeline behavior, "
        "and trainability on small models (3B dense / 7B hybrid). Give a short scored table. "
        "Then RECOMMEND ONE as the primary training criterion — OR, if justified, a "
        "principled COMPOSITE that takes the best-supported term from specific candidates "
        "(state exactly which term from which candidate, and argue why the composite is "
        "coherent and not a hedge). Write the final recommended criterion out in full LaTeX "
        "with all terms defined. End with one HONEST paragraph: what this criterion will "
        "genuinely train into the model, and what it cannot.\n\n"
        "=== THE THREE CANDIDATES ===\n" + cards
    )


def call_r1(prompt: str, label: str, max_tokens: int, temperature: float) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    for attempt in range(1, RETRIES + 1):
        t0 = time.time()
        try:
            r = httpx.post(URL, headers=headers, json=body, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            dt = time.time() - t0
            print(f"  [{label}] ok in {dt:.0f}s  tok_out={usage.get('completion_tokens','?')}", flush=True)
            if msg.strip():
                return msg.strip()
            print(f"  [{label}] empty body, retry ({attempt}/{RETRIES})", flush=True)
        except Exception as e:
            print(f"  [{label}] attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(3)
    return f"_(R1 failed on stage '{label}' after {RETRIES} attempts.)_"


def flush(title: str, text: str) -> None:
    with MASTER.open("a") as f:
        f.write(f"## {title}\n\n{text}\n\n---\n\n")


def main() -> None:
    print("=" * 70)
    print("DERIVE the trainable divergence criterion from the Constellax pipeline")
    print(f"model={MODEL}  out={MASTER}")
    print("=" * 70)

    MASTER.write_text(
        "# The Divergence Criterion — R1 derivation from the Constellax pipeline\n\n"
        f"_model: {MODEL}. Isolated derivation; provenance traced to real pipeline stages._\n\n"
        "> Goal: compress the pipeline's one conserved operation into a strict, trainable "
        "criterion for distilling divergent thinking into a small (Granite 3B / 7B) model. "
        "R1 was told to keep the divergence<->grounding<->consequence tension and to mark "
        "honestly what is proven vs asserted.\n\n---\n\n"
    )

    # Stage 1 — decipher + 3 candidate families
    print("\n-> STAGE 1: decipher + 3 candidate families ...", flush=True)
    decipher = call_r1(P_DECIPHER, "decipher", max_tokens=4096, temperature=0.5)
    (OUTDIR / "stage1_decipher.md").write_text(decipher)
    flush("Stage 1 — Decipher: the conserved operation + three candidate families", decipher)

    # Stage 2 — develop each candidate fully
    cards = []
    for i in (1, 2, 3):
        print(f"\n-> STAGE 2.{i}: develop candidate #{i} ...", flush=True)
        out = call_r1(p_candidate(i, decipher), f"cand{i}", max_tokens=8192, temperature=0.3)
        (OUTDIR / f"stage2_candidate{i}.md").write_text(out)
        flush(f"Stage 2 — Candidate #{i} (spec · genesis · proofs · fit · trainability)", out)
        cards.append(f"### CANDIDATE #{i}\n{out}")

    # Stage 3 — adjudicate
    print("\n-> STAGE 3: adjudicate + recommend ...", flush=True)
    adj = call_r1(p_adjudicate("\n\n".join(cards)), "adjudicate", max_tokens=6144, temperature=0.2)
    (OUTDIR / "stage3_adjudicate.md").write_text(adj)
    flush("Stage 3 — Adjudication: scored comparison + recommended criterion", adj)

    print("\n" + "=" * 70)
    print(f"DONE -> {MASTER}")
    print("=" * 70)


if __name__ == "__main__":
    main()
