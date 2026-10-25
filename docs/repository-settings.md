# Maintainer release settings

These settings describe the current development website. Package releases remain a separate action.

- Pages serves the root of the **gh-pages** branch. Source remains on `main`; generated
  HTML, search indexes and guide bundles are confined to the hosting branch.
- The Documentation workflow builds both MkDocs and the capability explorer, then updates
  `gh-pages` with a normal push and explicitly requests a Pages build. A workflow-token push
  alone does not trigger branch-based Pages publication. When account billing prevents a job from starting, run the
  same locked build locally; do not count the blocked job as a successful check.
- Add repository description: `Programmable chemical worlds for controlled, replayable agent experimentation.`
- Set the homepage to `https://sunyrain.github.io/ChemWorld-Public/`.
- Add topics: `ai4science`, `chemistry`, `agent-evaluation`, `gymnasium`, `world-model`, `reinforcement-learning`.
- Create releases by pushing a version-matching tag only after CI and Documentation both pass.

Local website publication after validation:

```bash
uv sync --locked --extra dev --extra docs
uv run --no-sync python scripts/build_site.py --output site
git fetch origin gh-pages:gh-pages
uv run --no-sync ghp-import --no-jekyll --branch gh-pages site
git push origin gh-pages
```

The integrated website is served at `/?lang=en` and `/?lang=zh`. Its original guide
reader uses `#guides/<capability>/<document>/<section>` routes, rendered from `docs/`
Markdown. The former `/explore/`, `/docs/` and published guide URLs redirect into
that interface, retaining language and section. Static MkDocs output remains a
fallback for technical pages outside the domain-guide collection.
The recordings are historical test snapshots, not a live backend.

The release workflow creates a GitHub Release and attaches source and Python package artifacts. PyPI
publication is deliberately not enabled until package ownership and trusted publishing are configured.
