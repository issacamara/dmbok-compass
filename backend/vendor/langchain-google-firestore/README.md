# Firestore LangChain compatibility package

Maintainer: DMBOK Compass implementation owner (the project sponsor until ownership is reassigned).

This is a narrow compatibility build of Google's `langchain-google-firestore` 0.5.0, distributed under Apache-2.0. The package code retains Google's copyright and license headers. The maintained source is the `langchain_google_firestore/` directory here; the adjacent wheel is a release artifact built from that source. Upstream project: <https://pypi.org/project/langchain-google-firestore/>. Check for a supported upstream release before updating this fork, and retire the fork once upstream supports the same LangChain core and score contract.

Changes from 0.5.0:

- Permit the tested LangChain 1.x and `langchain-community` 0.4.x dependency family.
- Fix `add_texts` to use the correct IDs and metadata after each 500-document batch, with a fresh Firestore batch per commit.
- Add `similarity_search_with_score` that asks Firestore for its native distance result and returns `1 - cosine_distance` alongside each `Document`. The requested corpus filter and result count are applied before vector search. The `0.5.0.post2` candidate uses a Firestore-valid distance field name; the earlier `post1` candidate failed this live check.

No application policy, collection activation, or citation logic belongs in this package. The compatibility checks in `backend/compat/` cover the modified behavior. A live development Firestore index check is required before the application adopts this package. Open an upstream issue or pull request for the batch and score fixes when network access is available, and record its link here.

Build from the repository root with Python 3.11+ and `setuptools`/`wheel` installed:

```sh
SOURCE_DATE_EPOCH=1760227200 python3 -m pip wheel --no-build-isolation --no-deps backend/vendor/langchain-google-firestore --wheel-dir /tmp/dmbok-firestore-wheel
shasum -a 256 /tmp/dmbok-firestore-wheel/langchain_google_firestore-0.5.0.post2-py3-none-any.whl
```

The expected wheel SHA-256 is `9ced9b491908b453a31a4b2598c1ec5fbbddf106dff868adcfe36bd5e60124c2`. Two independent builds produced this hash on 2026-10-12. Verify the vendored wheel against it before installation. `backend/requirements-compat.lock` pins the complete candidate runtime dependency set; it is not yet the deployed application lock.
