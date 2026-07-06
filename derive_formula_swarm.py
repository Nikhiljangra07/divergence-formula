"""
derive_formula_swarm.py — a SWARM of 10 web-enabled DeepSeek-R1 agents that take the
Constellax pipeline + the question as their shared foundation, roam ANY field (physics,
biology, game theory, mathematics, economics, ...) for inspiration and live verification,
and EACH construct one complete candidate criterion + proof.

GENERATE-ONLY by design: the swarm produces 10 distinct formulas + proofs and STOPS.
Policing (adversarial scrutiny) and synthesis are done by the human afterwards — this
script's only job is to hand over 10 well-formed, provenance-traced, self-proved
candidates for that review.

ISOLATED + SAFE. No production code touched. Reads only OPENROUTER_API_KEY. Writes only
into ./out/. Every agent result is flushed to its own file the moment it returns.
No time/token limit per agent (generous ceilings). The ONLY global limit is HARD_USD_CAP:
a running cost ledger refuses to launch new agents once spend would breach the cap.

WEB ACCESS: each agent uses OpenRouter's web plugin (live search, snippet depth) so it can
pull real information from any domain — not just its training weights.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derive_formula import API_KEY, MODEL, URL, SHARED_CONTEXT  # noqa: E402

OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)
MASTER = OUTDIR / "swarm_formulas.md"

# --- resources: generous; the ONLY hard limit is money -----------------------------
HARD_USD_CAP = 8.00
PRICE_IN = 0.55 / 1_000_000   # conservative OpenRouter deepseek-r1 input  $/token
PRICE_OUT = 2.50 / 1_000_000  # conservative output $/token (reasoning billed as output)
WEB_PER_CALL = 0.02           # OpenRouter web plugin ~ $4 / 1k results, 5 results/call
WEB_MAX_RESULTS = 5
CONCURRENCY = 8               # true parallel R1 agents in flight
TIMEOUT = 600.0             # no time pressure — 10 min ceiling per call
RETRIES = 3
MAXTOK_BUILD = 14000        # ample room to think, search, prove, and write

# --- the ten lenses: math + physics + biology, each free to roam further -----------
LENSES = [
    ("information_theory",
     "Information theory: mutual information, conditional entropy, channel capacity, "
     "rate-distortion. Divergence as low inter-thread mutual information; grounding as "
     "bounded distortion / KL to the problem anchor; consequence as a downstream channel."),
    ("determinantal_geometry",
     "Determinantal point processes & geometric volume: diversity as the log-determinant of "
     "a similarity kernel over thread embeddings (repulsion = volume); grounding as a "
     "per-thread anchor-proximity margin; consequence as an extra coordinate block."),
    ("game_decision_theory",
     "Game / decision theory: threads as strategies, consequences as payoff vectors, the user "
     "as the decision-maker who picks ex post; real-options value, regret, and an adversary "
     "that tries to collapse threads — equilibrium keeps them apart."),
    ("dynamical_variational",
     "Dynamical systems & variational calculus: a trajectory through reasoning state-space; "
     "divergence as positive Lyapunov spread / no contraction to a fixed point; grounding as "
     "staying in a basin around the problem; an action functional whose extremals are the "
     "open branch-set, never the single argmax."),
    ("optimal_transport",
     "Optimal transport: the thread set as a probability measure over option-space; divergence "
     "as Wasserstein/energy dispersion of that measure; grounding as a marginal/support "
     "constraint tying mass to the problem; consequence as a pushforward map."),
    ("statistical_mechanics",
     "Statistical mechanics (physics): a free energy F = E - T*S where S is thread entropy "
     "(divergence) and E an anchoring energy (grounding); temperature T tunes the balance; "
     "find the order parameter for the converge<->diverge phase boundary."),
    ("category_order_theory",
     "Category / order theory: a NON-collapsing functor — a coproduct (keep-separate) rather "
     "than a product/limit (merge); preserving an antichain of incomparable branches; "
     "convergence is the forbidden terminal-object collapse."),
    ("topology_persistence",
     "Topology / persistent homology: threads as connected components (H0) of a structure "
     "built near the problem; divergence as the count & persistence of distinct components; "
     "grounding as a bounded filtration radius from the anchor; collapse as premature merging."),
    ("evolutionary_ecology",
     "Evolutionary ecology (biology): niche differentiation, competitive exclusion, adaptive "
     "radiation; divergence as ecological diversity (Shannon/Simpson/phylogenetic) across "
     "thread-niches; grounding as shared-environment fitness; consequence as each lineage's "
     "projected survival. Premature convergence = competitive exclusion to a monoculture."),
    ("quantum_superposition",
     "Quantum theory (physics): the thread-set as a superposition of basis states held "
     "coherent until the USER's choice acts as measurement and collapses it; divergence as "
     "state distinguishability / coherence; grounding as the Hamiltonian (problem) the states "
     "live under; forced convergence = premature decoherence. Consequence as each eigenstate's "
     "measured outcome."),
]


class Ledger:
    def __init__(self, cap: float) -> None:
        self.cap, self.spent, self.calls = cap, 0.0, 0
        self.lock = asyncio.Lock()

    async def charge(self, usage: dict, web: bool) -> float:
        cost = (usage.get("prompt_tokens", 0) * PRICE_IN
                + usage.get("completion_tokens", 0) * PRICE_OUT
                + (WEB_PER_CALL if web else 0.0))
        async with self.lock:
            self.spent += cost
            self.calls += 1
        return cost

    def over(self) -> bool:
        return self.spent >= self.cap


LEDGER = Ledger(HARD_USD_CAP)
SEM = asyncio.Semaphore(CONCURRENCY)


async def r1_web(client: httpx.AsyncClient, prompt: str, label: str) -> str:
    if LEDGER.over():
        print(f"  [{label}] SKIPPED — money cap ${LEDGER.cap:.2f} reached "
              f"(spent ${LEDGER.spent:.2f})", flush=True)
        return "_(skipped — global money cap reached before this agent ran.)_"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK_BUILD,
        "temperature": 0.5,
        "plugins": [{"id": "web", "max_results": WEB_MAX_RESULTS}],  # live web search
    }
    async with SEM:
        for attempt in range(1, RETRIES + 1):
            t0 = time.time()
            try:
                resp = await client.post(URL, headers=headers, json=body, timeout=TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]["content"] or ""
                # web results show up as 'annotations' / url citations when the plugin fires
                used_web = "annotations" in (data["choices"][0]["message"] or {}) or True
                cost = await LEDGER.charge(data.get("usage", {}), used_web)
                dt = time.time() - t0
                print(f"  [{label}] ok {dt:.0f}s "
                      f"tok_out={data.get('usage',{}).get('completion_tokens','?')} "
                      f"+${cost:.3f} (total ${LEDGER.spent:.2f}/{LEDGER.cap:.0f})", flush=True)
                if msg.strip():
                    return msg.strip()
                print(f"  [{label}] empty body, retry {attempt}/{RETRIES}", flush=True)
            except Exception as e:
                print(f"  [{label}] attempt {attempt}/{RETRIES} {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(3)
    return f"_(R1 failed on '{label}' after {RETRIES} attempts.)_"


def build_prompt(lens_name: str, lens_guide: str) -> str:
    return SHARED_CONTEXT + "\n\nYOUR TASK NOW:\n" + (
        f"You are one of ten independent researchers. Your primary lens is **{lens_name}**:\n"
        f"{lens_guide}\n\n"
        "You have LIVE WEB SEARCH. Use it freely to roam ANY field — physics, biology, "
        "economics, game theory, evolutionary dynamics, control theory, anything — for "
        "structural inspiration AND to verify your claims against real results. Cite the "
        "sources you actually pull (URL or paper).\n\n"
        "First, in 2-3 lines, state YOUR reading of the single conserved operation the "
        "pipeline performs. Then construct ONE complete candidate criterion for distilling "
        "that operation into the model. Deliver:\n"
        "(1) FORMAL SPEC — objects, spaces, the criterion in LaTeX; every symbol defined; the "
        "divergence / grounding / consequence terms explicit and the tension encoded "
        "(constraint / Lagrangian / product / equilibrium — whatever is natural to your lens).\n"
        "(2) GENESIS — derive it line by line from the REAL pipeline; map each term to the "
        "specific stage it comes from (laundering, wander, governor CLOSE, coverage D_t, "
        "shepherd, blender distinct-thesis preservation, formalizer). No invented provenance.\n"
        "(3) PROOF(S) — prove what is provable: maximized iff threads are mutually distinct "
        "AND grounded; pure-divergence degenerates without the grounding term; the "
        "non-convergence (no-argmax) invariant is preserved; any bounds / monotonicity. Mark "
        "clearly proven vs asserted vs heuristic. Bring in real theorems (cite them) where "
        "they help.\n"
        "(4) FIT-TEST — does it describe what the pipeline actually does (wander expands, "
        "blender preserves distinct theses, nothing force-converged)? Where is the gap?\n"
        "(5) TRAINABILITY — operational as an SFT target / reward / corpus-acceptance filter on "
        "a 3B dense and a 7B hybrid model? Is each term computable at training time? Failure "
        "mode + how to detect it.\n"
        "Use LaTeX. Be rigorous and honest — a human will scrutinize and synthesize this "
        "afterwards, so make it auditable. End with a 6-line SUMMARY CARD "
        "(name; form; strongest provenance; strongest proof; biggest fit-gap; trainability; "
        "key sources)."
    )


def flush(title: str, text: str) -> None:
    with MASTER.open("a") as f:
        f.write(f"## {title}\n\n{text}\n\n---\n\n")


async def main() -> None:
    print("=" * 72)
    print(f"SWARM (generate-only) — {len(LENSES)} web-enabled R1 agents, 1 formula+proof each")
    print(f"model={MODEL} +web  hard money cap=${HARD_USD_CAP:.2f}  concurrency={CONCURRENCY}")
    print("=" * 72)

    MASTER.write_text(
        "# Ten Divergence Criteria — R1 SWARM (generate-only)\n\n"
        f"_model: {MODEL} + live web search. {len(LENSES)} independent agents, each one "
        "complete formula + proof, traced to real pipeline stages. Policing & synthesis are "
        f"done by the human afterwards. Hard money cap ${HARD_USD_CAP:.2f}; no time/token "
        "limit per agent._\n\n---\n\n"
    )

    async with httpx.AsyncClient() as client:
        print(f"\n== generating {len(LENSES)} candidates ==", flush=True)
        outs = await asyncio.gather(*[
            r1_web(client, build_prompt(name, guide), f"build:{name}")
            for name, guide in LENSES
        ])
        for (name, _), text in zip(LENSES, outs):
            (OUTDIR / f"swarm_formula_{name}.md").write_text(text)
            flush(f"Candidate — {name}", text)

    print("\n" + "=" * 72)
    print(f"DONE -> {MASTER}")
    print(f"agents run: {LEDGER.calls}   total spend: ${LEDGER.spent:.2f} / ${HARD_USD_CAP:.2f} cap")
    print("Next step is YOURS: police the 10, then synthesize.")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
