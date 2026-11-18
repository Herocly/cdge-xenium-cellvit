# Original source attribution and execution

`mapping_original.py` is the unchanged source shared by Emanuele Carelli on
Slack on 2026-10-01 and downloaded on 2026-10-02.

The B1.6 adapter compiles its function definitions verbatim and supplies the
offline CL graph instead of executing the top-level online `ms` initializer.
No original function body is patched, no first-match or tie rules are
replaced, and the original default four-class fallback is retained.

The existing input spelling/plural normalizer runs only after original
raw-name lookup fails; it does not alter any original tree/class function.
Necessary database-input/class-schema bridges are documented in the package
README and recorded in audits. Offset is zero, so no interactive merging is
performed. The original abundance data processing is not substituted for
CDGE's frozen marker-vote, seed, CNV or propagation algorithms.

No distribution license was supplied. Clarify permission before public
redistribution; retain the author credit.
