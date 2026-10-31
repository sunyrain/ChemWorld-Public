# Legacy query compatibility fixtures

These two Python snapshots are the exact original uncached/full-detail premixed and opposed implementations saved before the explicit numerical opt-ins. They are test compatibility oracles only. They must never be imported by production code, selected as a solver fallback, or used as independent scientific/reference evidence. Their shared imports intentionally resolve to the currently selected ChemWorld package.

The test loaders use private `chemworld_test_legacy_*` module names. Paths are relative to this fixture directory, so source-tree and isolated installed-package tests use the same fixtures. The fixture namespace is outside `chemworld`; no runtime codec, facade, cache or physical model is introduced.

`authored_opposed.py` is the exact authored synthetic input-factory function from the original opposed functional test, with only its imports separated. It avoids a dependency on test-collection order or cross-test-module imports under pytest importlib mode. It contains no real material coefficients, source phase or saved source state.

Snapshot SHA256 (compatibility identity, not a qualification certificate):

- gas_premixed_flow.py: 82a46c88f3e759aea6e62702ed84678dfdaa5db53cac213c87dfab771f9b655a
- gas_opposed_flow.py: e1dc2297a813d56a939c74326df9b33d72e79b7c3738a83e8092b81400e53162

Preserve these compatibility snapshots if the new implementation changes. Updating the snapshot merely to make a failing comparison pass is not a valid fix. New physics or API changes should get separately authored expectations and an explicit compatibility decision.
