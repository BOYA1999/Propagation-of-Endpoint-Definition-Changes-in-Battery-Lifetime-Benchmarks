# Security and privacy boundary

This release is intentionally anonymous and code-only. It contains no author name, affiliation, correspondence address, email address, ORCID, local user name, absolute local path, access token, password, API key, raw data, model weight, cache, notebook checkpoint, document metadata, or remote-repository credential.

Before public upload, run:

```powershell
python scripts/verify_public_repo.py
```

The check is deterministic against `MANIFEST.sha256`. If any file is added or changed intentionally, review it first and then regenerate the manifest. Do not add author metadata to the anonymous review branch. Add final authorship and repository citation metadata only in the public, author-approved release.

