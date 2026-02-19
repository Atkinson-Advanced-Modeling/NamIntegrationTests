# Reusable Workflow Guide

This document describes how to set up NamIntegrationTests as a **reusable workflow** that can be called from multiple repos (e.g., neural-amp-modeler, NeuralAmpModelerCore). This keeps integration test logic in a single place instead of duplicating it across repos.

## Overview

- **NamIntegrationTests**: Define a reusable workflow (with `workflow_call`) that runs the integration tests
- **Calling repos** (neural-amp-modeler, NeuralAmpModelerCore, etc.): Add a lightweight workflow that triggers on PRs and calls this repo's workflow

## 1. Create the Reusable Workflow in NamIntegrationTests

Create `.github/workflows/integration-tests.yml` in this repo:

```yaml
name: Integration Tests

on:
  workflow_call:
    inputs:
      source_repo:
        required: true
        type: string
        description: 'Repo to test (e.g., neural-amp-modeler or NeuralAmpModelerCore)'
      ref:
        required: false
        type: string
        default: ${{ github.sha }}
        description: 'Ref/branch to checkout'
    secrets:
      passphrase:
        required: false

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Checkout source repo
        uses: actions/checkout@v4
        with:
          repository: Atkinson-Advanced-Modeling/${{ inputs.source_repo }}
          ref: ${{ inputs.ref }}
          path: source
      # ... your actual integration test steps (install deps, run tests, etc.)
```

## 2. Create Caller Workflows in Each Repo

### In neural-amp-modeler

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
      source_repo: neural-amp-modeler
      ref: ${{ github.event.pull_request.head.sha }}
```

### In NeuralAmpModelerCore

Same pattern:

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

## 3. Permissions

- **Public repos**: Workflows can call reusable workflows in public repos without extra setup
- **Private repos**: Go to NamIntegrationTests → Settings → Actions → General → **Access**, and enable "Accessible from repositories owned by the user or organization"

## 4. Optional: Manual Trigger

Add `workflow_dispatch` to the reusable workflow so you can run tests manually from the Actions tab:

```yaml
on:
  workflow_call:
    # ... inputs
  workflow_dispatch:
    inputs:
      source_repo:
        required: true
        type: choice
        options:
          - neural-amp-modeler
          - NeuralAmpModelerCore
        description: 'Repo to test'
      ref:
        required: false
        type: string
        default: main
```

## 5. TODO: Flesh Out the Integration Test Steps

The `# ... your actual integration test steps` section needs to be implemented based on:

- What the integration tests actually do (checkout trainer? run pytest? etc.)
- Dependencies and environment setup (conda, pip, etc.)
- Any secrets (e.g., model weights, passphrase) that need to be passed through

## References

- [GitHub Docs: Reusing workflows](https://docs.github.com/en/actions/using-workflows/reusing-workflows)
- [GitHub Docs: workflow_call](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#workflow_call)
