# Public distribution

This repository publishes a selected development distribution of ChemWorld.
Install from this checkout with `uv sync --locked --extra dev`, or build wheel
and source archives with `uv build`. The source version remains `0.5.0.dev0`;
no new PyPI release is implied by a repository update.

The package contains the runtime, approved simulator resources, environment CLI,
Python examples and optional CPU build source. Research inputs, private evaluation
sets, credentials, provider payloads, internal UI prototypes and paper assets are
outside the distribution. The source-selection hook also supports rebuilding the
sdist without the research repository.

The only console command is `chemworld`. Historical benchmark applications have
the explicit [legacy entry point](legacy.md). Native workflows need no upstream
chemistry solver; optional reference libraries are used for separate comparisons.
The [current foundation page](current-foundation.md) records validation scope.

For old paper results, use the [frozen source and lockfile](https://github.com/sunyrain/ChemWorld-Public/tree/03e8026301c185fd6ba5bdbda7460765d9b3e724).
Current physics does not reproduce every historical trajectory.
