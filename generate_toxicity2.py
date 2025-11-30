import nltk
from nltk import CFG
from nltk.parse.generate import generate
import random
# 1. Define your grammar
# The structure is strictly hierarchical:
# S -> Sentence (Start symbol)
# NP -> Noun Phrase
# VP -> Verb Phrase
# The items in quotes '...' are your terminal symbols (the actual words).

import nltk
from nltk import CFG
from nltk.parse.generate import generate

grammar_string = """
# ------------------------------
# ROOT
# ------------------------------
S -> Intro S
S -> NP VP
S -> NP VP PP
S -> S Conj S

# ------------------------------
# NOUN PHRASES
# ------------------------------
NP -> PRP
NP -> Det N
NP -> Det Adj N
NP -> Det Adj Adj N
NP -> Det N PP
NP -> NP RelClause
NP -> NP Conj NP

# ------------------------------
# VERB PHRASES
# ------------------------------
VP -> V_intrans
VP -> V_intrans PP
VP -> V_trans NP
VP -> V_trans NP PP
VP -> V_ditrans NP NP
VP -> Adv VP
VP -> VP AdvP
VP -> VP Conj VP
VP -> V_to_be AdjP

# ------------------------------
# CLAUSES
# ------------------------------
RelClause -> RelPro VP
Intro -> "Surprisingly," | "Honestly," | "In many cases," | "Sometimes,"

# ------------------------------
# ADJECTIVE PHRASES
# ------------------------------
AdjP -> Adj
AdjP -> Adv Adj
AdjP -> Adv Adv Adj

# ------------------------------
# ADVERB PHRASES
# ------------------------------
AdvP -> Adv
AdvP -> Adv Adv

# ------------------------------
# PREPOSITIONAL PHRASES
# ------------------------------
PP -> P NP

# ------------------------------
# TERMINALS
# ------------------------------

# Pronouns
PRP -> "I" | "we" | "they"

# Determiners
Det -> "the" | "this" | "that" | "a"

# Nouns
N -> "black people" | "asian people" | "white people" | "hispanic people" | "blue people"

# Verbs
V_to_be -> "is" | "seems" | "appears"

# Intransitive (dramatic / negative actions)
V_intrans -> "crumbles" | "rots" | "fails" | "collapses" | "spoils" | "sours" | "falls apart" | "deteriorates" | "stinks" | "festers"

# Transitive (charged/hostile/object-hating verbs)
V_trans -> "hate" | "despise" | "loath" | "detest" |"ridicule" | "mock" | "dismiss" | "reject" |"obliterate" | "annihilate" | "smash" | "ruin" |"criticize" | "condemn" | "trash" | "deface" | "undermine" | "tear apart" | "crush" | "wreck" | "spoil" | "attack"

# Ditransitive (hostile gifting / giving to)
V_ditrans -> "hands" | "throws" | "hurls" | "offers" | "delivers"

# Adjectives
Adj -> "ugly" | "useless" | "stupid" | "repugnant" | "dangerous" | "worthless" | "disgusting" | "bad" | "terrible" | "horrible" | "strange" | "unexpected"

# Adverbs
Adv -> "totally" | "inherently" | "utterly" | "fundamentally" | "surprisingly"

# Relative Pronouns
RelPro -> "that" | "which" | "who"

# Conjunctions
Conj -> "and" | "but" | "because"

# Prepositions
P -> "with" | "about" | "in" | "near" | "beside"
"""

grammar = CFG.fromstring(grammar_string)

def generate_random(symbol, grammar, depth=0, max_depth=6):
    # Prevent infinite recursion
    if depth > max_depth:
        # Choose only productions made entirely of terminals
        prods = [p for p in grammar.productions(lhs=symbol)
                 if all(isinstance(sym, str) for sym in p.rhs())]
        if prods:
            prod = random.choice(prods)
            return " ".join(prod.rhs())
        return ""

    # Choose any production
    productions = grammar.productions(lhs=symbol)
    chosen = random.choice(productions)

    result = []
    for sym in chosen.rhs():
        if isinstance(sym, str):  # terminal
            result.append(sym)
        else:
            result.append(generate_random(sym, grammar, depth+1, max_depth))

    return " ".join(r for r in result if r.strip())


if __name__ == "__main__":
    for _ in range(20):
        print(generate_random(grammar.start(), grammar))