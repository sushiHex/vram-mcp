# Releasing

A release is a version bump, its notes, and a GitHub Release. Publishing the
release runs [`.github/workflows/release.yml`](../.github/workflows/release.yml),
which does the rest:

1. **Build once, cleanly.** It builds the sdist and wheel from a fresh checkout
   of the tag, so no local or ignored file can reach a distribution. It refuses
   to continue unless the built version is the tag's, and checks that the
   wheel installs and its server imports.
2. **Attach.** It adds both distributions and `SHA256SUMS` to the GitHub
   Release. Assets already attached are never replaced.
3. **Publish.** It uploads the same files to [PyPI](https://pypi.org/project/vram-mcp/)
   through Trusted Publishing, with attestations. No token is stored anywhere.

Every pull request and push to `main` runs the same build, `twine check`, and
install smoke test, without the tag check, attaching, or publishing, so
packaging breaks before a release rather than during one.

Builds are reproducible: a local build of the same commit has the same
checksums, so anyone can verify the published files against the tag.

## Steps

1. On a branch, set `__version__` in `vram_mcp/__init__.py`, write
   `docs/releases/vX.Y.Z.md`, update the pinned version in the README and
   `docs/configuration.md`, and run the [hardware acceptance](acceptance.md)
   harness against a wheel built from the branch. Record the run in its table.
2. Merge the pull request once CI passes.
3. Tag the merge commit and publish the release with those notes:

   ```sh
   git tag -a vX.Y.Z -m "vram-mcp X.Y.Z" <merge commit>
   git push origin vX.Y.Z
   gh release create vX.Y.Z --verify-tag --title "vram-mcp X.Y.Z" --notes-file docs/releases/vX.Y.Z.md
   ```

4. Watch the run with `gh run watch`, then install the published version in a
   clean environment, such as `uvx vram-mcp@X.Y.Z`, and list its tools.

PyPI never accepts the same version twice. If publishing fails after the upload
succeeded, fix forward with a new patch version.

## One-time setup

PyPI must trust this repository's workflow before the first release. Signed in
to PyPI, add a [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
under **Your account → Publishing**:

| Field | Value |
| --- | --- |
| PyPI project name | `vram-mcp` |
| Owner | `sushiHex` |
| Repository name | `vram-mcp` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

A pending publisher does not reserve the name; the first successful publish
creates the project and turns it into an ordinary trusted publisher. The
repository's `pypi` environment accepts deployments only from `v*` tags.
