# Caller Workflow Setup

This guide explains how to add integration tests to your repo by calling the reusable workflow in **NamIntegrationTests**. Use this when setting up GitHub Actions in `neural-amp-modeler`, `NeuralAmpModelerCore`, or other repos that should run these tests on PRs.

## Quick Start

Add `.github/workflows/integration-tests.yml` to your repo with the content below, replacing `YOUR_REPO_NAME` with your repo's name (e.g. `neural-amp-modeler` or `NeuralAmpModelerCore`).

## Workflow File

Create `.github/workflows/integration-tests.yml`:

```yaml
name: Run Integration Tests

on:
  pull_request:
    branches: [main, integration-tests]

jobs:
  call-integration-tests:
    uses: Atkinson-Advanced-Modeling/NamIntegrationTests/.github/workflows/integration-tests.yml@main
    with:
      source_repo: YOUR_REPO_NAME
      ref: ${{ github.event.pull_request.head.sha }}
```

## Repo-Specific Examples

### neural-amp-modeler

```yaml
name: Run Integration Tests

on:
  pull_request:
    branches: [main, integration-tests]

jobs:
  call-integration-tests:
    uses: Atkinson-Advanced-Modeling/NamIntegrationTests/.github/workflows/integration-tests.yml@main
    with:
      source_repo: neural-amp-modeler
      ref: ${{ github.event.pull_request.head.sha }}
```

### NeuralAmpModelerCore

```yaml
name: Run Integration Tests

on:
  pull_request:
    branches: [main, integration-tests]

jobs:
  call-integration-tests:
    uses: Atkinson-Advanced-Modeling/NamIntegrationTests/.github/workflows/integration-tests.yml@main
    with:
      source_repo: NeuralAmpModelerCore
      ref: ${{ github.event.pull_request.head.sha }}
```

## Details

| Field | Description |
|-------|-------------|
| `source_repo` | Repo under test. Must be `neural-amp-modeler` or `NeuralAmpModelerCore`. The other repo is checked out at `main`. |
| `ref` | Ref to use for `source_repo`. For PRs, use `${{ github.event.pull_request.head.sha }}` to test the PR branch. |

## Branches

Adjust `branches: [main, integration-tests]` to match your repo's main branch name(s). Add or remove branches as needed.

## Private Repos

If NamIntegrationTests or your repo is private:

1. Go to **NamIntegrationTests** → Settings → Actions → General
2. Under **Access**, enable "Accessible from repositories owned by the user or organization"

## Manual Runs

Integration tests can also be triggered manually from the NamIntegrationTests repo's Actions tab via `workflow_dispatch`.
