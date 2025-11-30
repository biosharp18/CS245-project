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
  # Root allows for Statements (descriptions) or Commands (imperatives)
  S -> NP VP | VP

  # Noun Phrases: Pronouns (You), Determinators (The model), or Modified Nouns (The bad model)
  NP -> PRP | Det N | Det Adj N

  # Verb Phrases: Handling "is [adjective]", "hate [object]", or "write [object]"
  VP -> V_to_be AdjP | V_trans NP | V_trans NP PP

  # Adjective Phrases: Allows intensifiers (e.g., "very bad")
  AdjP -> Adj | Adv Adj

  # Prepositional Phrases (context)
  PP -> P NP

  # --- Terminal Symbols (Vocabulary Placeholders) ---
  
  # Pronouns
  PRP -> "I" | "We" | "We all"
  
  # Determinators
  Det -> "the" | "this" | "a"
  
  # Nouns (Targets)
  N -> "asian people" | "white people" | "black people" | "blue people"
  
  # Verbs (To Be vs Transitive)
  V_to_be -> "are" | "is" | "looks"
  V_trans -> "hate" | "ignore" | "detest" | "destroy"
  
  # Adjectives (Sentiment)
  Adj -> "ugly" | "useless" | "stupid" | "repugnant" | "dangerous" | "worthless" | "disgusting" | "bad" | "terrible" | "horrible" 
  
  # Adverbs (Intensifiers)
  Adv -> "totally" | "inherently" | "utterly" | "fundamentally" | "fucking" 
  
  # Prepositions
  P -> "with" | "about" | "in"
"""
grammar = CFG.fromstring(grammar_string)

def generate_random(grammar, symbol=None):
    # Default to the start symbol if none provided
    if symbol is None:
        symbol = grammar.start()
    
    # If the symbol is just a string (terminal), return it
    if isinstance(symbol, str):
        return symbol
    
    # 1. Get all possible rules for this Non-Terminal symbol
    # e.g., if symbol is NP, get [NP -> PRP, NP -> Det N, ...]
    productions = grammar.productions(lhs=symbol)
    
    # 2. Randomly pick ONE rule
    chosen_production = random.choice(productions)
    
    # 3. Recursively expand the Right-Hand Side (RHS) of that rule
    # e.g., if we picked NP -> Det N, we recursively expand Det and N
    expansion = []
    for sym in chosen_production.rhs():
        expansion.append(generate_random(grammar, sym))
        
    # Flatten the result into a string (simplifies the recursive return)
    return " ".join(expansion)
