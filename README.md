# NamIntegrationTests

Integration tests for the [Neural Amp Modeler](https://www.neuralampmodeler.com) trainer and core repos.

## Repositories

- **Trainer** — [sdatkinson/neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) — Machine learning code for creating new models
- **Core** — [sdatkinson/NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) — Low-level DSP code for real-time model playback

## Setup

Clone the trainer and core as siblings of this repo:

```
parent_dir/
├── NamIntegrationTests/
├── neural-amp-modeler/
└── NeuralAmpModelerCore/
```

1. **Trainer** (required): Install neural-amp-modeler, e.g. `pip install -e ../neural-amp-modeler`, or add it to `PYTHONPATH` when running tests.
2. **Core** (required for loadmodel tests): Build the core's loadmodel tool:
   ```bash
   cd ../NeuralAmpModelerCore
   cmake -B build && cmake --build build
   ```
3. **This project**: `pip install -e .`

## Running tests

```bash
pytest test/
```

If the trainer is not installed via pip, add it to `PYTHONPATH`:

```bash
PYTHONPATH=../neural-amp-modeler pytest test/
```

Tests that require the core's `loadmodel` tool are skipped if NeuralAmpModelerCore is missing or loadmodel is not built.
