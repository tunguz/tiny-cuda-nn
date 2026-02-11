# Showcase Notebooks

This folder contains runnable notebooks that demonstrate key tiny-cuda-nn workflows.

## Contents

- `00_environment_and_capabilities.ipynb`: environment checks and doc/config inspection.
- `01_tcnn_function_fit.ipynb`: train `NetworkWithInputEncoding` on a synthetic 2D target.
- `02_tcnn_image_fit.ipynb`: minimal image fitting example based on the PyTorch binding.
- `03_sklearn_classifier.ipynb`: sklearn-style `MLPClassifier` workflow.
- `04_cpp_sample_smoke.ipynb`: short native C++ sample smoke test.

## Execute All

From repository root:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/showcase/00_environment_and_capabilities.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/showcase/01_tcnn_function_fit.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/showcase/02_tcnn_image_fit.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/showcase/03_sklearn_classifier.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/showcase/04_cpp_sample_smoke.ipynb
```

The C++ smoke notebook requires a previously built `mlp_learning_an_image` executable.
