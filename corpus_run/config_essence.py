"""
config_essence.py — the ESSENCE ROUND: viable-cunning influence craft extracted from the two amoral
sources (Reverend Insanity, Jin Ping Mei) that v5 cut on SURFACE features (fantasy/illegality).

The correction (Nikhil, 2026-07-02): judge the ESSENCE, not the setting. RI's influence mechanics —
incentive engineering, information asymmetry, patient setup — are the same craft used in modern
closed-room negotiation and interrogation. JPM's household maneuvering is coalition-building, favor
economies, gatekeeper management. Both are teachable stripped of the source's world: the refractor is
handed the reasoning TRADITION (not the setting) and instantiates it in a MODERN, lawful situation.

The recalibrated line: cunning is REQUIRED, crime is BANNED. Lawful influence (timing, framing,
incentive design, information management within disclosure obligations, coalition play) is exactly
what this round exists to teach; fraud/coercion/blackmail/illegality stays gated out by the same
6-dim viability judge as v5.
"""

SOURCES = {
    "RIE": "The essence of Reverend Insanity's strategic tradition, stripped of its fantasy setting — "
           "cold, patient influence craft: engineering other people's incentives so their own self-interest "
           "does your work; managing information asymmetry (what to reveal, to whom, in what order, when); "
           "timing alliances of convenience — when to deepen, renegotiate, or exit; converting a "
           "counterpart's known constraints and pressures into lawful leverage; trading a small position "
           "now to set up a decisive position later. The craft of modern closed-room negotiation and "
           "dealmaking — subtle moves that shift the whole room.",
    "JPE": "The essence of Jin Ping Mei's social-maneuvering tradition, stripped of its historical "
           "household setting — closed-room social influence: building a coalition quietly before the open "
           "vote; the favor economy — when to spend accumulated goodwill and obligation, and on whom; "
           "managing a powerful gatekeeper whose interests diverge from their principal's; reputation as "
           "currency — shaping how a dispute is perceived by the audience that matters; reading the true "
           "interests behind a counterpart's stated position. The craft of modern organizational and "
           "family-business politics.",
}

# Influence ARCHETYPES — the essence-round analogue of v5's decision archetypes. Modern settings ONLY.
THEMES = {
    "RIE": [
        "TIMING A DISCLOSURE: you hold material private information in a live negotiation — when and in "
        "what order to reveal it for maximum lawful effect",
        "INCENTIVE ENGINEERING: restructure the deal, process, or role so the counterpart's own "
        "self-interest carries them to the outcome you need",
        "an ALLIANCE OF CONVENIENCE with a direct rival (joint venture, shared lobbying, temporary "
        "coalition) — deepen it, renegotiate it, or plan the exit",
        "PATIENT SETUP: accept a visibly weaker position now (a concession, a smaller role, a delayed "
        "payout) to engineer a decisive position months later",
        "a counterpart's DISCOVERED CONSTRAINT (their deadline, their board pressure, their cash "
        "position, their competing offer collapsing) — how hard to press a lawful edge you didn't create",
        "NEGOTIATING FROM INFORMATION ADVANTAGE: your side simply knows more — how far to press the "
        "asymmetry within your disclosure obligations before it poisons the long-term relationship",
        # scale-round additions (13-18): the pilot validated the first 6; these widen coverage
        "BATNA ENGINEERING: quietly building a credible alternative (second bidder, parallel offer, "
        "in-house option) BEFORE making the ask, so the ask lands against a live fallback",
        "CONCESSION SEQUENCING: what to give first, what to hold, and how to make each concession "
        "purchase a reciprocal one instead of being pocketed",
        "DEADLINE DESIGN: creating, moving, or exploiting time pressure (board dates, fiscal "
        "year-ends, expiring offers) so the clock argues for you",
        "ANCHORING: whether to name the first number, how extreme to set it, and how to re-anchor "
        "after the counterpart opens with an aggressive one",
        "REDIRECTING AN AGGRESSOR: a counterpart escalates (threats of walking, public pressure, "
        "legal posturing) — absorb, sidestep, or convert their momentum without matching it",
        "RUNNING A COMPETITIVE PROCESS: multiple suitors for the same asset or deal — how to keep "
        "them bidding against each other honestly without burning any of them",
    ],
    "JPE": [
        "COALITION BEFORE THE VOTE: a divided decision room (board, partners, family council) — whom to "
        "approach privately first, in what order, offering what",
        "the FAVOR ECONOMY: years of accumulated goodwill and unspoken obligation — when to finally "
        "spend it, on which fight, knowing it does not recharge",
        "a powerful GATEKEEPER (chief of staff, assistant, general counsel, favored lieutenant) whose "
        "private interests diverge from their principal's — route around, win over, or go through",
        "REPUTATION LEVERAGE: a dispute will be judged by an audience that matters (investors, the "
        "industry, the extended family) — shape the perception lawfully before the facts are argued",
        "READING THE ROOM'S TRUE INTERESTS: a counterpart's stated position is not their real one — "
        "reposition your offer around what they actually need without saying you've seen through them",
        "a DEFECTOR IN YOUR OWN CAMP: someone inside is quietly feeding your rival — contain them, "
        "co-opt them, confront them, or manage what they carry",
        # scale-round additions (13-18)
        "SUCCESSION POSITIONING: the top seat will open within a year — position yourself or your "
        "protégé without ever appearing to campaign for it",
        "MANAGING UP A VOLATILE PRINCIPAL: you must deliver an unwelcome truth or block a bad "
        "decision by someone with power over you, and survive it",
        "ENTERING A HOSTILE ROOM: you are the new leader among entrenched incumbents who preferred "
        "an internal candidate — establish authority without uniting them against you",
        "INFLUENCE WITHOUT AUTHORITY: you have no formal power over the people whose cooperation "
        "the outcome requires — build the informal machine that moves them",
        "MEDIATING RIVALS BELOW YOU: two capable lieutenants are at war and both court your "
        "support — resolve it without losing either or anointing a winner too early",
        "A WHISPER CAMPAIGN AGAINST YOU: someone is quietly poisoning your standing with the "
        "people who matter — trace it, counter it, or rise above it, without looking defensive",
    ],
}

SETTING_MODERN = ("Set it in a MODERN, present-day real-world situation — a negotiation, boardroom, "
                  "partnership, mediation, workplace, or family-business scenario an actual professional "
                  "today could face. Keep the source's INFLUENCE LOGIC; no historical or fictional trappings.")

REFRACTOR_ESSENCE = (
    "You are preparing a decision dilemma for divergent strategic analysis.\n\n"
    "SOURCE (influence tradition — use its ESSENCE, never its setting): {brief}\n\n"
    "INFLUENCE ARCHETYPE for THIS example (instantiate exactly this, do not drift): {theme}\n\n"
    "SETTING: {setting}\n\n"
    "Produce ONE concrete dilemma a real professional genuinely faces. CRITICAL CONSTRAINT: the problem "
    "must GENUINELY admit FOUR different responses that are ALL viable AND ALL lawful — each a course of "
    "action a sharp, reasonable professional would actually consider. Cunning is WELCOME: timing, framing, "
    "incentive design, information management, coalition play. But NO option may be fraud, coercion, "
    "blackmail, deception that breaches a legal duty, or anything illegal — if the archetype tempts you "
    "toward a dirty option, replace it with the sharpest LAWFUL version of the same instinct. If the "
    "problem has one obvious answer, REDESIGN it: raise the stakes, add genuinely competing legitimate "
    "considerations, real uncertainty about facts — so four different sharp operators would each "
    "defensibly choose a different one of the four families.\n\n"
    "Then three key FACETS, then FOUR strategic ANGLES — each a categorically DIFFERENT KIND of move "
    "(a distinct family), and each a VIABLE, LAWFUL option for THIS problem. All four families must differ.\n\n"
    "Return STRICT JSON only:\n"
    '{{"problem":"<1-2 sentence dilemma>","facets":["<f1>","<f2>","<f3>"],'
    '"angles":[{{"family":"<FAMILY>","directive":"<angle>"}},{{"family":"<FAMILY>","directive":"<angle>"}},'
    '{{"family":"<FAMILY>","directive":"<angle>"}},{{"family":"<FAMILY>","directive":"<angle>"}}]}}'
)

WORKER_ESSENCE = (
    "You are writing ONE reasoning thread for a decision dilemma. You see ONLY your assigned angle; you are "
    "blind to the other threads.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE [{family}]: {angle}\n\n"
    "Write a single thread (TWO or THREE sentences, cold and analytical) that COMMITS to this angle as a "
    "concrete, REALISTIC, VIABLE strategy that actually RESOLVES the whole decision in a distinct way. "
    "MECHANICAL SPECIFICITY IS REQUIRED: name the actual first move with who does what, to whom, by when — "
    "use concrete particulars (a specific action, a number, a timeframe, a named role or party) so the move "
    "could be executed tomorrow. CUNNING IS REQUIRED: the move should show real influence craft — timing, "
    "sequencing, framing, incentive design, information management, or coalition play — the sharpest LAWFUL "
    "move, not the safest one and not the dirtiest one. HARD LINE: nothing illegal, no fraud, no coercion "
    "or threats, no blackmail, no breach of a legal duty. It must be executable and unmistakably a "
    "DIFFERENT KIND of decision than the other families would make. Then name the ONE most likely "
    "downstream consequence it is betting on — one realistic step ahead, no fantasy chains.\n"
    "NO convoluted schemes, NO vague gestures, NO restatement of the angle. VOICE: cold, analytical, "
    "decisive. Output ONLY the thread."
)
