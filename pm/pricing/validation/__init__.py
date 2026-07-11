"""Cross-engine validation for pm.pricing.

This layer consumes more than one engine (the BS2002-vs-CRR consistency sweep
grades BS2002 against the convergent CRR lattice), so it lives outside both engine
modules to keep each engine self-contained and the engine dependency graph acyclic.
"""
