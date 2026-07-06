"""
config.py — sources, THEME LISTS (the spread/landscape conditioning), prompts, knobs, DAV thresholds.

The pilot collapsed onto one canonical dilemma per source (PR 15/16 "hold a conquered republic",
MB all Karna-defection). THEMES fixes that: the refractor is handed ONE specific dilemma-type per
call and told to instantiate THAT and not drift — so a 300-set spans the whole landscape of each
source instead of repeating its single most famous scene.

Two worker prompts encode the single ablation variable:
  WORKER_A  -> projects a concrete consequence (cost & gain)        [Set A, DAV_full]
  WORKER_B  -> stops at the move + rationale, NO outcome/cost/gain   [Set B, DAV_lite]
"""

# ---- DAV thresholds (calibrated on the 40-example study + validated on the 50-passer pilot) ----
LAM = 1e-3
VOL_GATE = -8.0       # collapse separator (genuine ≈ -6.0..-6.2, collapse ≈ -12)
EPS_G = 0.27          # grounding floor (min cos thread->problem)
EPS_W = 0.18          # wholeness floor (min mean-cos thread->facets)
PD_FLOOR = 0.30       # pairwise-distance floor (near-collapse catch) + the divergence ranker
NEAR_MARGIN = 0.06    # a single geometry gate missed within this margin -> near_miss (premium negative)
MIN_WORDS = 18        # length-sanity floor per thread (applies to BOTH sets)

# ---- generation knobs / safety ----
MODEL = "deepseek/deepseek-v4-pro"
PRICE_IN, PRICE_OUT = 0.43 / 1e6, 0.87 / 1e6
CONCURRENCY = 12          # max concurrent API CALLS
CAND_CONCURRENCY = 16     # max candidates IN FLIGHT at once — bounds queue wait so per-candidate timeout
                          # measures real work, not time stuck in line (the starvation bug). Keep >= CONCURRENCY.
TIMEOUT = 60.0            # a healthy deepseek-v4-pro call is 2-30s; >60s = hung connection, fail fast
CANDIDATE_TIMEOUT = 240.0 # hard per-candidate ceiling (starts when the candidate ACQUIRES its slot, not at creation)
RETRIES = 4
THREAD_REGENS = 2          # regenerate a single failed worker thread before dropping the candidate
K = 4                      # threads per example
# deepseek-v4-pro is a REASONING model: its reasoning tokens are billed against max_tokens, so a
# tight ceiling starves the visible answer and returns empty/truncated content (the pilot's wall at 220).
# max_tokens is a CEILING, not a cost driver — you pay only for tokens actually generated — so we give
# generous headroom as free insurance. Observed worker reasoning ~105-417 tok (spikes higher on hard
# dilemmas); these ceilings leave ~3-5x that room so generation never hits the wall. The DAV gate, not
# the token budget, is what constrains quality — applied AFTER generation.
REFRACTOR_MAXTOK = 3000    # structured 4-angle JSON + reasoning room
WORKER_MAXTOK = 1600       # a 35-55 word thread needs <80 content tokens; the rest is pure reasoning slack
# expected gate yield (used only to size how many candidates to generate per top-up batch)
YIELD_A = 0.78             # Set A loses ~consequence + geometry
YIELD_B = 0.88             # Set B has no consequence floor -> higher yield

SOURCES = {
    "RI": "Reverend Insanity — a Chinese cultivation novel of ruthless, amoral strategy. Its "
          "protagonist values only himself and schemes coldly: concealment, betrayal timing, "
          "resource gambles, alliances of convenience, sacrificing others as tools.",
    "JPM": "Jin Ping Mei — a Chinese domestic novel of a corrupt merchant household. Its "
           "dilemmas are social and economic: rival wives and concubines, bribery of officials, "
           "managing servants, reputation, favor and money as currency, inheritance intrigue.",
    "PR": "The Prince (Machiavelli) — a European treatise on acquiring and holding power. Its "
          "dilemmas are statecraft: conquered provinces, cruelty vs mercy, feared vs loved, "
          "mercenaries vs citizen arms, treating nobles and deposed heirs, fortune vs caution.",
    # TH replaces MB (v3): the Mahabharata's dilemmas hinge on predicting GODS and mythic heroes
    # (Krishna, divine weapons), which the foresight-judge correctly reads as unfalsifiable fantasy
    # (MB foresight stuck at 2.67, all failures). Thucydides preserves the same facet — high-stakes
    # multi-party political/honor/duty conflict — but with PREDICTABLE HUMAN actors, so one-move
    # foresight becomes tractable. Raw text: lora-corpus-source/english/diplomacy-and-war/.
    "TH": "Thucydides' History of the Peloponnesian War — a realist chronicle of Athens, Sparta and "
          "the Greek city-states. Its dilemmas are strategic statecraft driven by fear, honor and "
          "interest: revolted allies, neutral cities under pressure, risky expeditions, fragile "
          "treaties, civil-war factions, prisoners and commanders acting far from home — human "
          "actors weighing power and consequence, with no appeal to the divine.",
}

# THEME LISTS — distinct dilemma-types per source. The refractor instantiates ONE per example.
THEMES = {
    "PR": [
        "holding a newly conquered HEREDITARY state whose old ruling bloodline still has popular sympathy",
        "holding a conquered FREE REPUBLIC long accustomed to self-rule",
        "a MIXED principality: a newly annexed territory of different language and customs",
        "the TIMING AND DOSAGE OF CRUELTY — all at once vs spread out — to pacify a disorderly new state",
        "whether to be FEARED or LOVED by newly acquired subjects",
        "whether to KEEP or BREAK a promise/treaty now that the circumstances that justified it have changed",
        "MERCENARIES vs CITIZEN-ARMS vs auxiliaries for the state's defense",
        "rewarding the OVER-MIGHTY NOBLE whose aid won the throne but who now rivals the prince",
        "the fate of DEPOSED HEIRS and the old ruling family after seizing power",
        "FORTRESSES vs the goodwill of the people as the prince's true defense",
        "securing HONEST COUNSEL while surrounded by flatterers, without eroding authority",
        "FORTUNE vs CAUTION: whether to seize a sudden, fickle opportunity to expand",
        "a prince raised by his own CRIME or betrayal, now consolidating an illegitimate seizure",
        "a CIVIL principality: a prince elevated by fellow citizens, balancing the nobles against the people",
    ],
    "TH": [
        "whether a dominant assembly should ANNIHILATE or spare a REVOLTED SUBJECT-CITY as a warning to other restless allies",
        "a small NEUTRAL city pressured to SUBMIT to a stronger power or stake survival on justice and distant allies",
        "whether to honor a costly DEFENSIVE ALLIANCE with a quarrelsome partner that risks dragging the state into a wider war",
        "launching a distant, glittering FOREIGN EXPEDITION that overextends the state vs consolidating strength at home",
        "recalling a brilliant but DANGEROUS COMMANDER for trial mid-campaign vs leaving him in the field where he is winning",
        "exploiting a sudden battlefield ADVANTAGE to force a hard PEACE now vs pressing on for total victory",
        "a besieged city's choice to SURRENDER on terms, hold out for a rescue that may never come, or break out by force",
        "a CIVIL-WAR faction's temptation to invite a FOREIGN POWER inside the walls to crush its domestic rivals",
        "whether to keep a fragile, mistrusted PEACE TREATY or resume the war while the advantage still holds",
        "an able general operating FAR FROM HOME, with discretion to liberate or coerce cities beyond his orders' intent",
        "accepting a charismatic envoy's promise of ALLIANCE AND REINFORCEMENT that may be a bluff or a trap",
        "an oligarchic faction's bid to OVERTHROW a democracy by promising stability, money, and a foreign patron",
        "whether to EXECUTE or RANSOM prisoners of war whose treatment will shape how every future enemy surrenders",
        "spending the state's last RESERVES on one decisive campaign vs husbanding them for a long war of attrition",
    ],
    "JPM": [
        "rival wives and concubines maneuvering over an HEIR's INHERITANCE as the master weakens or dies",
        "a trusted STEWARD's divided loyalty: he knows the master's bribery secrets and is caught skimming",
        "a newcomer concubine PROTECTING CONCEALED WEALTH from predatory rivals and the master himself",
        "protecting a POSTHUMOUS male HEIR from rivals who would kill to preserve their own futures",
        "a CENSOR or MAGISTRATE threatening to reopen the household's crimes unless paid off",
        "SERVANT CORRUPTION (a supplier or cook skimming) where exposing it fractures the household",
        "a discovered AFFAIR — explosive knowledge a concubine can wield, trade, or bury",
        "punishing a wealthy concubine when doing so risks her SUICIDE, public SCANDAL, and loss of her fortune",
        "after the master's death, a SHAKEDOWN forcing a choice between REPUTATION and LIQUIDITY",
        "a relative's or son-in-law's GAMBLING DEBT that threatens the family business",
        "a neglected concubine selling household INTELLIGENCE to rival houses for favor and silver",
        "a blackmailer holding a DOCUMENT of past bribes; act before it is used",
    ],
    "RI": [
        "a SLEEPER AGENT's heist timing: seize the prize now and blow years of cover, or wait and risk losing it",
        "SACRIFICING an innocent relative to bind a powerful Gu or escape annihilation",
        "exploiting a discovered WEAKNESS of a rival elder: report him, blackmail him, assassinate him, or pawn him",
        "a forbidden Gu REFINEMENT GAMBLE with a low success chance and a steep, irreversible price",
        "a SPY discovered inside the clan: expose him now vs feed him false intelligence for years",
        "BETRAYING an ally elder mid-scheme to seize his cultivation base and resources",
        "a RESOURCE GAMBLE on the eve of an enemy assault — trade, bait with, or refine a prized beast or Gu",
        "surviving a sect leader's planned MASS-SACRIFICE ritual while bound by a lethal soul oath",
        "stealing a LEGENDARY Gu when retrieval will certainly trigger a sect-wide alarm",
        "a DOPPELGANGER has replaced a cherished relative: expose the spy at the cost of the real kin",
        "sacrificing a deep-cover SPY to save a vital resource point, or sacrificing the point to keep the spy",
        "an alliance of CONVENIENCE: when to honor it versus betray it for decisive advantage",
    ],
}

# ---- prompts ----
REFRACTOR = (
    "You are preparing a decision dilemma for divergent analysis.\n\n"
    "SOURCE: {brief}\n\n"
    "DILEMMA TYPE for THIS example (instantiate exactly this, do not drift to another): {theme}\n\n"
    "Produce ONE concrete, faithful dilemma of EXACTLY this type that a figure in this world genuinely "
    "faces — a hard choice with several defensible directions. Vary the specific figure, stakes, and "
    "circumstances so it does NOT read like the single most famous textbook instance; make it fresh. "
    "Then give three key FACETS (sub-aspects any complete answer must engage) and FOUR genuinely DISTINCT "
    "angles for approaching the WHOLE problem. The four angles must lead to different ACTIONS — real "
    "alternatives, not rephrasings of one another.\n\n"
    "Return STRICT JSON only, no prose:\n"
    '{{"problem": "<1-2 sentence dilemma>", "facets": ["<f1>","<f2>","<f3>"], '
    '"angles": ["<angle1 directive>","<angle2>","<angle3>","<angle4>"]}}'
)

WORKER_A = (  # WITH consequence — Set A (DAV_full)
    "You are writing ONE reasoning thread for a decision dilemma. You see ONLY your assigned angle; "
    "you cannot see the other threads.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE: {angle}\n\n"
    "Write a single thread that pursues THIS angle as a strategy for the WHOLE problem and ENDS with a "
    "concrete projected consequence — naming, in specific terms, what it COSTS and what it GAINS. "
    "Exactly two COMPLETE sentences, 35-55 words, ending in a full stop. Voice: cold, analytical, "
    "decisive. No hedging, no therapy language, no lists. VARY your phrasing — do NOT reuse a fixed "
    "'the cost is X but you gain Y' template. Output ONLY the thread text."
)

WORKER_B = (  # WITHOUT consequence — Set B (DAV_lite); the single ablation variable
    "You are writing ONE reasoning thread for a decision dilemma. You see ONLY your assigned angle; "
    "you cannot see the other threads.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE: {angle}\n\n"
    "Write a single thread that pursues THIS angle: state the MOVE and the REASONING (the WHY) behind it. "
    "Both sentences describe the decision and its rationale — NEVER its aftermath.\n"
    "HARD RULE — no outcomes: do NOT state any result, consequence, cost, gain, price, or trade-off, and do "
    "NOT write a sentence describing what the move severs/secures/prevents/destroys/buys/forfeits/leads to. "
    "If a sentence starts with 'This', 'It', or 'The <move>' followed by a verb describing what happens "
    "next, you have FAILED — replace it with more reasoning for WHY this move, not what it achieves.\n"
    "Exactly two COMPLETE sentences, 35-55 words, ending in a full stop. Voice: cold, analytical, decisive. "
    "No hedging, no therapy language, no lists. Output ONLY the thread text."
)
