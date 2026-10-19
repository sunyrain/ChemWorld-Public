# World foundations

> Current main: [October foundation update](current-foundation.md). Existing figures and numerical evidence below describe the frozen August runtime.

ChemWorld separates what an experimental agent may observe from what an experiment author may control. That separation makes causal world families possible without turning hidden evaluator state into an answer key.

## The public contract

Every task exposes typed operations, parameter bounds, instruments, resource limits, termination rules and evaluation-facing outputs. The contract is stable enough for an agent, planner or RL policy to inspect before acting.

## The composed world

World authors combine reaction, thermal, phase, separation, crystallization, distillation, flow, electrochemical and observation components through a compatibility-checked composition contract. A valid world must satisfy the registered interfaces before an episode starts.

## Controlled changes

Parent and child worlds can differ in one registered law while holding the public experiment fixed. This supports a precise question: can an agent notice that an old local model no longer predicts its observations, and choose a better next intervention?

## Research directions

The stable release enables future studies of repeated experimentation, belief updating, informative intervention selection and mechanism-level multiple-choice judgments. These are research directions, not claims about unpublished experiments or results.

<div class="cw-status-note">Status: the software substrate and finite qualification evidence are public; ongoing experimental matrices, intermediate results and private evaluator materials are not part of this release.</div>

Start with [the controlled-world notebook](notebooks.md) or read the [world composition contract](world-composition-contract.md).
