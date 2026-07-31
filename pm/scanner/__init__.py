"""Scanner service layer — the deep-dive scanner drawer's compute side.

Four modules, one intra-package direction (``slice`` ← ``candidates`` ←
``ticket`` / ``view``): ``slice`` owns the sanctioned on-demand option-chain
pull and the state-attached caches; ``candidates`` generates, prices and
ranks roll / overlay / joint-roll adjustments; ``ticket`` composes the
adjustment ticket; ``view`` packages the drawer's read-only view data.
Every entry point takes the loaded ``PortfolioState`` explicitly — this
package owns no singleton and imports no Dash. ``pm.ui.state_access``
remains the UI's one seam: it resolves the runtime singleton once per call
and delegates here through same-signature wrappers. (The one ``pm.ui``
reference is the ticket's lazy import of the Dash-free
``pm.ui.deepdive.structure_economics`` leg-slice helpers.)
"""
