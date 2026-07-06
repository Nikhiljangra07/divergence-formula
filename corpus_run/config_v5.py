"""
config_v5.py — v5 "big corpus" source palette: 8 sources rebalanced AWAY from amoral-baroque scheming
(cut Reverend Insanity + Jin Ping Mei) TOWARD clear, viable, ethical decision-making.

3 strategic (Sun Tzu, Thucydides, Plutarch) + 4 ethical/human (Cicero, Aristotle, Dostoevsky, Stoics)
+ 1 reduced realism anchor (The Prince). Themes are DECISION-ARCHETYPES — abstract enough to instantiate
in a CLASSICAL or a MODERN setting (the generator alternates), which is what gives the corpus its variety
and fixes the weak modern domains (career/health/finance/relationship/ethics) the benchmark exposed.
"""
from config import MODEL, PRICE_IN, PRICE_OUT  # reuse generator model + pricing

SOURCES = {
    "ST": "Sun Tzu's Art of War — the classic of strategic economy. Winning before fighting, knowing when "
          "to engage vs withdraw, deception, positioning, decisive use of limited force. Every choice is a "
          "clear calculated commitment, never a convoluted scheme.",
    "TH": "Thucydides' History of the Peloponnesian War — realist statecraft driven by fear, honor and "
          "interest: revolted allies, neutral cities under pressure, risky expeditions, fragile treaties. "
          "Human actors weighing power and consequence, no appeal to the divine.",
    "PL": "Plutarch's Lives — Greek and Roman leaders at their decisive moments. Character-in-action: how a "
          "leader's judgment, ambition, and restraint shape a hard public choice, weighed by its real "
          "consequences for the leader and the state.",
    "CI": "Cicero's On Duties — the classical test for when the HONORABLE and the EXPEDIENT pull apart. The "
          "truly advantageous and the right rarely conflict once examined; every choice is weighed against "
          "duty, justice, and long-term reputation. The antidote to clever short-term scheming.",
    "AR": "Aristotle's Nicomachean Ethics — practical wisdom (phronesis): the right action is the mean "
          "between excess and deficiency, fitted to the particular case, aimed at a life well-lived. "
          "Deliberative and balanced, never extreme, never formulaic.",
    "DO": "Dostoevsky's moral novels — decisions under psychological pressure, guilt, and competing claims "
          "of conscience, faith, and survival. Every choice carries inner consequence; the easy "
          "rationalization is always the trap.",
    "ST2": "Stoic practical philosophy (Marcus Aurelius, Epictetus) — deciding by separating what is in "
           "your control from what is not, acting on duty and reason, accepting outcomes without being "
           "ruled by fear or desire. Calm, clarifying, aimed at right action under what cannot be changed.",
    "PR": "The Prince (Machiavelli) — acquiring and holding power: conquered provinces, cruelty vs mercy, "
          "feared vs loved, treating nobles and deposed rivals, fortune vs caution. Cold realism about "
          "power, used here as one voice among many.",
}

THEMES = {
    "ST": [
        "whether to ENGAGE a stronger rival now or maneuver to avoid the contest until conditions favor you",
        "committing limited resources to one DECISIVE strike vs preserving strength for a longer campaign",
        "when to use DECEPTION (appear weak or strong) vs deal openly with a counterparty",
        "seizing a sudden opening that OVEREXTENDS you vs holding your position",
        "choosing the GROUND: fight on your strength or deny the opponent theirs",
        "winning WITHOUT fighting — co-opting or dissuading a rival vs direct confrontation",
        "a cornered opponent: leave them an exit or press for total victory",
        "when to CUT LOSSES and withdraw from a failing engagement vs reinforce it",
        "acting on PARTIAL intelligence now vs waiting for certainty that may come too late",
        "concentrating force on one front vs dividing to threaten several",
        "exploiting an opponent's impatience or anger vs staying disciplined",
        "a first-mover gambit vs letting the rival commit first and countering",
        "allocating scarce attention across rival threats — which to meet, which to ignore",
        "revealing a capability now for DETERRENCE vs concealing it for surprise",
    ],
    "TH": [
        "whether a dominant party should DESTROY or spare a defector as a warning to others who might follow",
        "a smaller player pressured to SUBMIT to a stronger one or stake survival on justice and distant allies",
        "honoring a costly ALLIANCE with a quarrelsome partner that risks dragging you into a wider fight",
        "a distant, glittering EXPEDITION that overextends you vs consolidating strength at home",
        "recalling a brilliant but DANGEROUS lieutenant mid-campaign vs leaving them where they are winning",
        "exploiting a sudden ADVANTAGE to force a hard settlement now vs pressing for total victory",
        "a besieged position: SURRENDER on terms, hold for rescue that may not come, or break out",
        "a faction tempted to invite an OUTSIDE power in to crush its internal rivals",
        "keeping a fragile, mistrusted TRUCE vs resuming the fight while advantage holds",
        "an agent FAR FROM HOME with discretion to act beyond the intent of their orders",
        "accepting a charismatic envoy's promise of SUPPORT that may be a bluff",
        "whether to make an example of PRISONERS in a way that shapes every future surrender",
        "spending the last RESERVES on one decisive push vs husbanding them for a long contest",
        "a public assembly swept by fear or anger — ride the mood or resist it",
    ],
    "PL": [
        "a leader at the PEAK of success: press the advantage or consolidate and stop",
        "accepting an HONOR that breeds envy vs declining it for long-term standing",
        "punishing a victorious but INSUBORDINATE subordinate vs tolerating them",
        "a reformer's choice: push sweeping change FAST vs incremental, against entrenched interests",
        "honoring a costly PRINCIPLE publicly vs compromising for the institution's survival",
        "accepting RETREAT or exile to fight another day vs staking everything on a last stand",
        "grooming a capable RIVAL-in-waiting vs sidelining them",
        "CLEMENCY vs severity toward a defeated faction, knowing both carry cost",
        "taking personal CREDIT and risk vs sharing it to bind allies",
        "resisting the corrupting pull of POWER at the height of one's career",
        "spending political CAPITAL on a fight now vs banking it for later",
        "whether to TRUST a former enemy who now offers alliance",
        "a public figure's response to SLANDER: confront, ignore, or out-maneuver it",
        "choosing between GLORY (high risk and reward) and SECURITY (steady but modest)",
    ],
    "CI": [
        "how to handle a chance to PROFIT by a deception that harms no one obviously",
        "keeping a costly PROMISE whose circumstances have changed vs breaking it for gain",
        "whether to EXPOSE a friend's or colleague's wrongdoing — loyalty vs justice",
        "an obligation to FAMILY vs a duty to the wider community",
        "profiting from another's IGNORANCE in a deal vs full disclosure",
        "taking CREDIT owed to someone else when it would advance you",
        "accepting a BENEFIT that quietly compromises your independence",
        "punishing a wrong PROPORTIONATELY vs leniency vs letting it go",
        "whether to BEND a rule for a good outcome — the ends-justify-means test",
        "competing HARD against a rival vs a restraint that costs you the win",
        "a WINDFALL that legally belongs to you but morally to another",
        "whether to speak an UNWELCOME truth to someone with power over you",
        "personal ADVANCEMENT vs a commitment to a mentor or team",
        "honoring a contract's LETTER vs its spirit when they diverge",
    ],
    "AR": [
        "COURAGE vs recklessness vs cowardice: how much risk to take on a venture",
        "how GENEROUS to be with limited means — liberality vs waste vs stinginess",
        "when AMBITION becomes excess: how high to aim in a career",
        "honesty's mean: when CANDOR helps vs when tact serves better",
        "responding to insult: the mean between RAGE and spinelessness",
        "balancing WORK against the other goods of a life",
        "choosing FRIENDS or partners: utility vs pleasure vs shared character",
        "how much to INDULGE vs restrain a strong appetite or desire",
        "PRIDE vs humility: claiming the standing you've earned without excess",
        "when persistence is virtue vs when it becomes STUBBORNNESS",
        "SPENDING vs saving for the sake of a flourishing life",
        "weighing a present PLEASURE against a long-term good",
        "JUSTICE in dividing a shared reward among unequal contributors",
        "deciding under genuine UNCERTAINTY where no rule applies — the role of judgment",
    ],
    "DO": [  # reframed binary->multi-path ("how to navigate X") so 4 families fork: face / contain / involve-others / reframe
        "how to handle having committed a WRONG that would destroy you if it surfaced",
        "how to act when a 'GREATER GOOD' tempts you to cross a line you'd normally hold",
        "how to deal with someone who deeply BETRAYED you but is still in your life",
        "how to respond to a family member's heavy NEED when you may not owe it",
        "how to take up RESPONSIBILITY for a harm you partly caused",
        "how to handle knowing a TRUTH whose telling would wound people you care about",
        "how to respond to an UNJUST accusation or undeserved blame",
        "how to navigate a clash between LOYALTY to a sibling or parent and your own conscience",
        "how to respond to someone close who is DESTROYING themselves",
        "how to handle an offer of HELP that carries a moral string",
        "how to deal with a HYPOCRITE who holds power over you",
        "how to answer a grave WRONG done to you or someone you love",
        "how to navigate a failing COMMITMENT you took on out of duty",
        "how to decide whether and how to TRUST again after being broken",
    ],
    "ST2": [
        "a setback OUTSIDE your control: change course or endure and adapt",
        "chasing an EXTERNAL good (status, wealth) vs investing in what you control",
        "how to act when you cannot change a situation but must still choose your STANCE",
        "confronting a FEAR that's distorting a decision vs deferring to it",
        "acting on PRINCIPLE when it costs you vs staying silent",
        "managing a relationship with someone whose BEHAVIOR you can't change",
        "accepting a LIMIT or diagnosis and deciding how to live within it",
        "whether to keep STRIVING for a goal or release it as outside your power",
        "responding to UNFAIR treatment: protest, withdraw, or transcend",
        "how much present COMFORT to sacrifice for a duty you believe in",
        "what to do after a sudden loss of ROLE or identity (job, status)",
        "choosing your RESPONSE to another's provocation",
        "planning elaborately for an UNCERTAIN future vs acting on the present duty",
        "SIMPLIFYING a life that has become overextended",
    ],
    "PR": [
        "holding a newly acquired GROUP or territory whose old loyalty runs to a displaced predecessor",
        "the TIMING AND DOSAGE of a harsh but necessary measure — all at once vs spread out",
        "whether to be FEARED or LOVED by those you now lead",
        "keeping or BREAKING a commitment now that the circumstances that justified it have changed",
        "relying on HIRED help vs your own people for a critical capability",
        "rewarding an OVER-MIGHTY ally whose aid won you the position but who now rivals you",
        "the fate of a DEPOSED rival and their loyalists after you take over",
        "FORTRESSES vs goodwill — what actually secures your position",
        "securing HONEST counsel while surrounded by flatterers, without losing authority",
        "FORTUNE vs caution: whether to seize a sudden, fickle opportunity to expand",
        "consolidating a position won by a STAINED or aggressive move",
        "balancing two factions whose support you both NEED and fear",
    ],
}
