# sklearn-style bindings

The `tinycudann_sklearn` package exposes a scikit-learn inspired API for
`tiny-cuda-nn`.  It currently ships an `MLPClassifier` class that mirrors the
behaviour of `sklearn.neural_network.MLPClassifier` while executing the heavy
lifting with PyTorch (and optionally `tinycudann` when a compatible GPU build
is available).

## Usage

```python
import numpy as np
from tinycudann_sklearn import MLPClassifier

X = np.random.randn(1_000, 8).astype(np.float32)
y = np.random.randint(0, 4, size=1_000)

clf = MLPClassifier(
    hidden_layer_sizes=(64, 64, 64),
    max_iter=50,
    random_state=42,
    learning_rate_init=1e-2,
)

clf.fit(X, y)
print("accuracy:", clf.score(X, y))
```

The classifier exposes the familiar `fit`, `partial_fit`, `predict`,
`predict_proba`, `predict_log_proba`, `decision_function`, `score`,
`get_params`, and `set_params` methods.  When CUDA is available and a
`tinycudann` extension has been installed, homogeneous hidden layer shapes
automatically opt into the high-performance fused MLP kernels.
