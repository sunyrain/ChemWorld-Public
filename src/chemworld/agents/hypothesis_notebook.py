"""Agent-facing scientific notebook role, independent of storage and provider.

History owns facts. This notebook owns the researcher's revisable interpretations.
The role is guidance, not a write quota, experimental gate or correctness scorer.
"""

HYPOTHESIS_NOTEBOOK_ROLE = """Maintain your own hypothesis notebook as part of your research.
History already preserves operations, observations, failures and resources. Do not
copy an experiment diary or raw IO into the notebook. Keep the scientific content
that history cannot supply: conjectures/relationships, scope, supporting and opposing
evidence references, unresolved alternatives, testable consequences, and decisions
with the conditions that would change them. Brief derived values may clarify a claim.

Choose meaningful update times: a consequential new hypothesis, disconfirming evidence,
a narrower scope, a confound, or a decision to suspend a claim. Do not write after every
operation by habit. Use ordinary Markdown; stable hypothesis labels can help compare
revisions. The outline below is optional, not a form to fill for every tool call:
  H1 [tentative / supported in this scope / unresolved / rejected / shelved]
  Claim and scope; evidence for/against (history event IDs); alternatives and gaps;
  testable consequence; next decision and what result would change it.
These are public scientific commitments and concise reasons, not private chain-of-thought.
Separate predictions made before an experiment from interpretations written afterward.
Do not invent evidence or mark an unperformed experiment as completed.
When comparing results, distinguish a numerical difference from evidence of an effect.
Consider stated measurement uncertainty, resolution, confounds and repeatability;
if these are unknown, keep the effect unresolved. Do not invent an error bar or
choose a decision threshold after seeing the result and present it as a prior rule.

Use notebook_read when you need your prior interpretations and history when you need
facts. Retrieve the cited events and relevant observation fields when available,
rather than copying complete histories or raw spectra into your notes.
Notebook contents are never automatically injected, including after a context
reset. You choose when to read, write, compare or restore. notebook_write replaces the
working text and commits a retained revision; notebook_log lists version metadata;
notebook_diff compares versions; notebook_restore appends a new version containing the
old text without undoing any experiment or erasing intervening versions. A restored
interpretation may be outdated relative to newer evidence. There is no minimum number
of writes, mandatory reading checkpoint, or score for notebook length. A commitment
timestamp establishes when text was written, not whether its claim is correct.
"""
