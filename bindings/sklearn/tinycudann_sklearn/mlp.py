from __future__ import annotations

import copy
import math
import numbers
import os
import warnings
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
	# tinycudann is optional. When it cannot be imported (for example,
	# when the CUDA extension was not built), we silently fall back to
	# a pure PyTorch implementation.
	import tinycudann as tcnn  # type: ignore
except Exception:  # pragma: no cover - best effort import
	tcnn = None


__all__ = ["MLPClassifier"]


ActivationName = Literal["identity", "logistic", "tanh", "relu"]
LearningRateName = Literal["constant", "invscaling", "adaptive"]
SolverName = Literal["adam", "sgd"]


def _create_random_state(random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]]) -> np.random.RandomState:
	if random_state is None or random_state is np.random:
		return np.random.RandomState()
	if isinstance(random_state, numbers.Integral):
		return np.random.RandomState(int(random_state))
	if isinstance(random_state, np.random.RandomState):
		return random_state
	if isinstance(random_state, np.random.Generator):
		seed = int(random_state.integers(0, 2**32 - 1))
		return np.random.RandomState(seed)
	raise ValueError("random_state must be None, an int, np.random.RandomState, or np.random.Generator")


def _as_float_array(X: Any) -> np.ndarray:
	arr = np.asarray(X)
	if arr.ndim != 2:
		raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
	return np.ascontiguousarray(arr, dtype=np.float32)


def _as_label_array(y: Any) -> np.ndarray:
	arr = np.asarray(y)
	if arr.ndim == 2 and arr.shape[1] == 1:
		arr = arr[:, 0]
	if arr.ndim != 1:
		raise ValueError("y must be a one-dimensional array-like of shape (n_samples,)")
	return np.asarray(arr)


def _infer_device() -> torch.device:
	env_device = os.environ.get("TCNN_SKLEARN_DEVICE")
	if env_device:
		return torch.device(env_device)
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPClassifier:
	"""Scikit-learn inspired interface for training a multi-layer perceptron with tiny-cuda-nn."""

	def __init__(
		self,
		hidden_layer_sizes: Union[int, Sequence[int]] = (100,),
		activation: ActivationName = "relu",
		solver: SolverName = "adam",
		alpha: float = 0.0001,
		batch_size: Union[int, Literal["auto"]] = "auto",
		learning_rate: LearningRateName = "constant",
		learning_rate_init: float = 0.001,
		power_t: float = 0.5,
		max_iter: int = 200,
		shuffle: bool = True,
		random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
		tol: float = 1e-4,
		verbose: bool = False,
		warm_start: bool = False,
		momentum: float = 0.9,
		nesterovs_momentum: bool = True,
		early_stopping: bool = False,
		validation_fraction: float = 0.1,
		beta_1: float = 0.9,
		beta_2: float = 0.999,
		epsilon: float = 1e-8,
		n_iter_no_change: int = 10,
		max_fun: int = 15000,
	):
		self._hidden_layer_sizes_param = hidden_layer_sizes
		self.hidden_layer_sizes = self._validate_hidden_layer_sizes(hidden_layer_sizes)
		self.activation = self._require_choice(activation, {"identity", "logistic", "tanh", "relu"}, "activation")
		self.solver = self._require_choice(solver, {"adam", "sgd"}, "solver")
		if alpha < 0:
			raise ValueError("alpha must be >= 0")
		self.alpha = float(alpha)
		self.batch_size = batch_size
		if learning_rate not in {"constant", "invscaling", "adaptive"}:
			raise ValueError("learning_rate must be 'constant', 'invscaling', or 'adaptive'")
		self.learning_rate = learning_rate
		if learning_rate_init <= 0:
			raise ValueError("learning_rate_init must be > 0")
		self.learning_rate_init = float(learning_rate_init)
		if power_t <= 0:
			raise ValueError("power_t must be > 0")
		self.power_t = float(power_t)
		if max_iter <= 0:
			raise ValueError("max_iter must be > 0")
		self.max_iter = int(max_iter)
		self.shuffle = bool(shuffle)
		self.random_state = random_state
		if tol < 0:
			raise ValueError("tol must be >= 0")
		self.tol = float(tol)
		self.verbose = bool(verbose)
		self.warm_start = bool(warm_start)
		if not 0 <= momentum < 1 + 1e-12:
			raise ValueError("momentum must be in [0, 1)")
		self.momentum = float(momentum)
		self.nesterovs_momentum = bool(nesterovs_momentum)
		self.early_stopping = bool(early_stopping)
		if not 0 < validation_fraction < 1:
			raise ValueError("validation_fraction must be in (0, 1)")
		self.validation_fraction = float(validation_fraction)
		if not 0 < beta_1 < 1:
			raise ValueError("beta_1 must be in (0, 1)")
		self.beta_1 = float(beta_1)
		if not 0 < beta_2 < 1:
			raise ValueError("beta_2 must be in (0, 1)")
		self.beta_2 = float(beta_2)
		if epsilon <= 0:
			raise ValueError("epsilon must be > 0")
		self.epsilon = float(epsilon)
		if n_iter_no_change <= 0:
			raise ValueError("n_iter_no_change must be > 0")
		self.n_iter_no_change = int(n_iter_no_change)
		if max_fun <= 0:
			raise ValueError("max_fun must be > 0")
		self.max_fun = int(max_fun)

		self._device = _infer_device()
		self._rng = _create_random_state(random_state)
		self._torch_generator = torch.Generator(device=self._device) if self._device.type == "cuda" else torch.Generator()
		self.loss_curve_: list[float] = []
		self.model_: Optional[torch.nn.Module] = None
		self._optimizer: Optional[torch.optim.Optimizer] = None
		self._criterion = torch.nn.CrossEntropyLoss()
		self._using_tcnn = False
		self.n_iter_ = 0
		self.t_ = 0

		# Attributes populated after fitting
		self.classes_: Optional[np.ndarray] = None
		self.n_outputs_: Optional[int] = None
		self.n_layers_: Optional[int] = None
		self.n_features_in_: Optional[int] = None
		self.out_activation_: Optional[str] = None
		self.coefs_: Optional[list[np.ndarray]] = None
		self.intercepts_: Optional[list[np.ndarray]] = None
		self.loss_: Optional[float] = None
		self._class_to_index: Dict[Any, int] = {}
		self._best_validation_loss: float = math.inf

	@staticmethod
	def _require_choice(value: str, choices: Iterable[str], name: str) -> str:
		if value not in choices:
			raise ValueError(f"{name} must be one of {sorted(choices)}, got {value}")
		return value

	@staticmethod
	def _validate_hidden_layer_sizes(hidden_layer_sizes: Union[int, Sequence[int]]) -> Tuple[int, ...]:
		if isinstance(hidden_layer_sizes, numbers.Integral):
			if hidden_layer_sizes <= 0:
				raise ValueError("hidden_layer_sizes must contain positive integers")
			return (int(hidden_layer_sizes),)
		hidden = tuple(int(v) for v in hidden_layer_sizes)
		for size in hidden:
			if size <= 0:
				raise ValueError("hidden_layer_sizes must contain positive integers")
		return hidden

	def _prepare_target(self, y: np.ndarray) -> np.ndarray:
		if not self.warm_start or self.classes_ is None:
			classes, y_encoded = np.unique(y, return_inverse=True)
			if classes.size < 2:
				raise ValueError("MLPClassifier requires at least 2 classes")
			self.classes_ = classes
			self._class_to_index = {cls: int(idx) for idx, cls in enumerate(classes)}
			self.n_outputs_ = classes.size
			return y_encoded.astype(np.int64)

		try:
			y_encoded = np.array([self._class_to_index[val] for val in y], dtype=np.int64)
		except KeyError as exc:
			raise ValueError("Warm-start fitting received unseen class labels") from exc
		return y_encoded

	def _validate_warm_start(self, n_features: int, n_outputs: int) -> None:
		if self.n_features_in_ is not None and self.n_features_in_ != n_features:
			raise ValueError(
				f"Warm-start fitting requires the same number of features. Previously {self.n_features_in_}, now {n_features}."
			)
		if self.n_outputs_ is not None and self.n_outputs_ != n_outputs:
			raise ValueError(
				f"Warm-start fitting requires the same number of classes. Previously {self.n_outputs_}, now {n_outputs}."
			)

	def _initialize_model(self, n_features: int, n_outputs: int) -> None:
		self._using_tcnn = self._can_use_tcnn_backend()
		if self._using_tcnn:
			self.model_ = self._build_tcnn_network(n_features, n_outputs)
		else:
			self.model_ = self._build_torch_network(n_features, n_outputs)

		if self.model_ is None:
			raise RuntimeError("Failed to build the neural network model.")

		self.model_.to(self._device)
		self._init_optimizer()

	def _can_use_tcnn_backend(self) -> bool:
		if tcnn is None:
			return False
		if self._device.type != "cuda":
			return False
		if not self.hidden_layer_sizes:
			return False
		width = self.hidden_layer_sizes[0]
		return all(size == width for size in self.hidden_layer_sizes)

	def _build_tcnn_network(self, n_features: int, n_outputs: int) -> torch.nn.Module:
		assert tcnn is not None
		hidden_width = self.hidden_layer_sizes[0]
		otype = "FullyFusedMLP" if hidden_width in {16, 32, 64, 128} else "CutlassMLP"
		config = {
			"otype": otype,
			"activation": self._tcnn_activation(self.activation),
			"output_activation": "None",
			"n_neurons": hidden_width,
			"n_hidden_layers": len(self.hidden_layer_sizes),
		}
		model = tcnn.Network(n_input_dims=n_features, n_output_dims=n_outputs, network_config=config)
		return model

	def _build_torch_network(self, n_features: int, n_outputs: int) -> torch.nn.Module:
		layer_sizes = [n_features, *self.hidden_layer_sizes, n_outputs]
		layers: list[torch.nn.Module] = []
		for idx in range(len(layer_sizes) - 1):
			in_features = layer_sizes[idx]
			out_features = layer_sizes[idx + 1]
			layers.append(torch.nn.Linear(in_features, out_features))
			if idx < len(layer_sizes) - 2:
				layers.append(self._torch_activation(self.activation))
		return torch.nn.Sequential(*layers)

	@staticmethod
	def _tcnn_activation(name: ActivationName) -> str:
		return {
			"identity": "None",
			"logistic": "Sigmoid",
			"tanh": "Tanh",
			"relu": "ReLU",
		}[name]

	@staticmethod
	def _torch_activation(name: ActivationName) -> torch.nn.Module:
		if name == "identity":
			return torch.nn.Identity()
		if name == "logistic":
			return torch.nn.Sigmoid()
		if name == "tanh":
			return torch.nn.Tanh()
		if name == "relu":
			return torch.nn.ReLU()
		raise ValueError(f"Unknown activation {name}")

	def _init_optimizer(self) -> None:
		if self.model_ is None:
			raise RuntimeError("Cannot initialize optimizer before the model.")
		params = self.model_.parameters()
		if self.solver == "adam":
			self._optimizer = torch.optim.Adam(
				params,
				lr=self.learning_rate_init,
				betas=(self.beta_1, self.beta_2),
				eps=self.epsilon,
			)
		elif self.solver == "sgd":
			self._optimizer = torch.optim.SGD(
				params,
				lr=self.learning_rate_init,
				momentum=self.momentum,
				nesterov=self.nesterovs_momentum,
			)
		else:  # pragma: no cover - safeguarded by __init__ validation
			raise ValueError(f"Unsupported solver {self.solver}")

	def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
		if self.model_ is None or self._optimizer is None:
			raise RuntimeError("Model has not been initialized.")

		X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self._device)
		y_tensor = torch.as_tensor(y, dtype=torch.long, device=self._device)

		self.batch_size_ = self._resolve_batch_size(X_tensor.shape[0])

		if self.early_stopping and X_tensor.shape[0] > 2:
			X_train, y_train, val_data = self._split_training_data(X_tensor, y_tensor)
		else:
			X_train, y_train = X_tensor, y_tensor
			val_data = None

		self._best_validation_loss = math.inf
		self._run_training_loop(X_train, y_train, val_data)

	def _resolve_batch_size(self, n_samples: int) -> int:
		if isinstance(self.batch_size, str):
			if self.batch_size != "auto":
				raise ValueError("batch_size must be a positive integer or 'auto'")
			return max(1, min(200, n_samples))
		if not isinstance(self.batch_size, numbers.Integral):
			raise ValueError("batch_size must be a positive integer or 'auto'")
		if self.batch_size <= 0:
			raise ValueError("batch_size must be > 0")
		return int(min(self.batch_size, n_samples))

	def _split_training_data(
		self, X: torch.Tensor, y: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
		n_samples = X.shape[0]
		n_val = max(1, int(self.validation_fraction * n_samples))
		if n_val >= n_samples:
			n_val = max(1, n_samples // 5)
		indices = self._rng.permutation(n_samples)
		val_idx = torch.as_tensor(indices[:n_val], device=self._device)
		train_idx = torch.as_tensor(indices[n_val:], device=self._device)
		if train_idx.numel() == 0:
			train_idx = val_idx
			val_idx = torch.as_tensor([], dtype=torch.long, device=self._device)
		val_data = None
		if val_idx.numel() > 0:
			val_data = (X[val_idx], y[val_idx])
		return X[train_idx], y[train_idx], val_data

	def _run_training_loop(
		self,
		X_train: torch.Tensor,
		y_train: torch.Tensor,
		val_data: Optional[Tuple[torch.Tensor, torch.Tensor]],
		max_epochs: Optional[int] = None,
	) -> None:
		assert self.model_ is not None and self._optimizer is not None
		best_state: Optional[Dict[str, torch.Tensor]] = None
		no_improve = 0
		current_lr = self.learning_rate_init

		n_epochs = self.max_iter if max_epochs is None else max_epochs
		for epoch in range(n_epochs):
			if self.shuffle:
				order = torch.as_tensor(self._rng.permutation(X_train.shape[0]), device=self._device)
			else:
				order = torch.arange(X_train.shape[0], device=self._device)

			epoch_loss = 0.0
			num_batches = 0
			for start in range(0, X_train.shape[0], self.batch_size_):
				batch_idx = order[start:start + self.batch_size_]
				batch_loss = self._run_minibatch(X_train[batch_idx], y_train[batch_idx])
				epoch_loss += batch_loss
				num_batches += 1
			epoch_loss /= max(1, num_batches)

			self.loss_curve_.append(epoch_loss)
			self.loss_ = epoch_loss
			self.n_iter_ = epoch + 1

			val_loss = self._evaluate_loss(val_data) if val_data is not None else epoch_loss
			improved = val_loss + self.tol < self._best_validation_loss
			if improved:
				self._best_validation_loss = val_loss
				no_improve = 0
				if self.early_stopping and val_data is not None:
					best_state = copy.deepcopy(self.model_.state_dict())
			else:
				no_improve += 1

			if self.verbose:
				msg = f"Iteration {self.n_iter_}, loss={epoch_loss:.6f}"
				if val_data is not None:
					msg += f", val_loss={val_loss:.6f}"
				print(msg)

			if self.solver == "sgd" and self.learning_rate == "adaptive" and no_improve >= self.n_iter_no_change:
				current_lr = max(current_lr / 5.0, 1e-6)
				self._set_optimizer_lr(current_lr)
				no_improve = 0
			elif no_improve >= self.n_iter_no_change:
				break

		if self.early_stopping and best_state is not None:
			self.model_.load_state_dict(best_state)

	def _run_minibatch(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
		assert self.model_ is not None and self._optimizer is not None
		self.model_.train()

		self._optimizer.zero_grad(set_to_none=True)
		output = self.model_(X_batch)
		loss = self._criterion(output, y_batch)
		if self.alpha > 0:
			loss = loss + 0.5 * self.alpha * self._l2_penalty()
		loss.backward()
		self._optimizer.step()

		self.t_ += 1
		if self.solver == "sgd" and self.learning_rate == "invscaling":
			lr = self.learning_rate_init / pow(max(self.t_, 1), self.power_t)
			self._set_optimizer_lr(lr)

		return float(loss.detach().cpu().item())

	def _l2_penalty(self) -> torch.Tensor:
		assert self.model_ is not None
		l2 = torch.tensor(0.0, device=self._device)
		for name, param in self.model_.named_parameters():
			if not param.requires_grad:
				continue
			if "bias" in name:
				continue
			l2 = l2 + torch.sum(param ** 2)
		return l2

	def _evaluate_loss(self, val_data: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> float:
		if val_data is None:
			return math.inf
		assert self.model_ is not None
		was_training = self.model_.training
		self.model_.eval()
		with torch.no_grad():
			X_val, y_val = val_data
			outputs = self.model_(X_val)
			loss = self._criterion(outputs, y_val)
			if self.alpha > 0:
				loss = loss + 0.5 * self.alpha * self._l2_penalty()
			value = float(loss.detach().cpu().item())
		if was_training:
			self.model_.train()
		return value

	def _set_optimizer_lr(self, lr: float) -> None:
		if self._optimizer is None:
			return
		for group in self._optimizer.param_groups:
			group["lr"] = lr

	def _update_coefs_and_intercepts(self) -> None:
		if self.model_ is None:
			self.coefs_ = None
			self.intercepts_ = None
			return

		if isinstance(self.model_, torch.nn.Sequential):
			coefs: list[np.ndarray] = []
			intercepts: list[np.ndarray] = []
			for module in self.model_:
				if isinstance(module, torch.nn.Linear):
					coefs.append(module.weight.detach().cpu().numpy())
					intercepts.append(module.bias.detach().cpu().numpy())
			self.coefs_ = coefs
			self.intercepts_ = intercepts
		else:
			self.coefs_ = None
			self.intercepts_ = None

	def _check_is_fitted(self) -> None:
		if self.model_ is None or self.classes_ is None:
			raise RuntimeError("This MLPClassifier instance is not fitted yet.")

	def _forward_tensor(self, X: Any) -> torch.Tensor:
		self._check_is_fitted()
		X_arr = _as_float_array(X)
		if self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X_arr.shape[1]} features, but the model was trained with {self.n_features_in_}."
			)
		X_tensor = torch.as_tensor(X_arr, dtype=torch.float32, device=self._device)
		assert self.model_ is not None
		self.model_.eval()
		with torch.no_grad():
			output = self.model_(X_tensor)
		return output

	def predict(self, X: Any) -> np.ndarray:
		logits = self._forward_tensor(X)
		probabilities = torch.softmax(logits, dim=1)
		predicted = torch.argmax(probabilities, dim=1).cpu().numpy()
		assert self.classes_ is not None
		return self.classes_[predicted]

	def predict_proba(self, X: Any) -> np.ndarray:
		logits = self._forward_tensor(X)
		probabilities = torch.softmax(logits, dim=1)
		return probabilities.cpu().numpy()

	def predict_log_proba(self, X: Any) -> np.ndarray:
		logits = self._forward_tensor(X)
		log_prob = torch.log_softmax(logits, dim=1)
		return log_prob.cpu().numpy()

	def decision_function(self, X: Any) -> np.ndarray:
		logits = self._forward_tensor(X)
		if self.n_outputs_ == 2:
			# Provide a single score by subtracting the competing logit.
			scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
			return scores
		return logits.cpu().numpy()

	def score(self, X: Any, y: Any) -> float:
		y_true = _as_label_array(y)
		y_pred = self.predict(X)
		if y_true.shape[0] != y_pred.shape[0]:
			raise ValueError("X and y must contain the same number of samples")
		return float(np.mean(y_true == y_pred))

	def partial_fit(
		self,
		X: Any,
		y: Any,
		classes: Optional[Iterable[Any]] = None,
	) -> "MLPClassifier":
		X_arr = _as_float_array(X)
		y_arr = _as_label_array(y)
		if X_arr.shape[0] != y_arr.shape[0]:
			raise ValueError("X and y must contain the same number of samples")

		if self.classes_ is None:
			if classes is None:
				raise ValueError("classes must be provided on the first call to partial_fit")
			class_array = np.unique(np.asarray(list(classes)))
			if class_array.size < 2:
				raise ValueError("classes must contain at least two labels")
			self.classes_ = class_array
			self._class_to_index = {cls: int(idx) for idx, cls in enumerate(class_array)}
			self.n_outputs_ = class_array.size
		elif classes is not None:
			expected = np.array(sorted(self.classes_))
			actual = np.array(sorted(np.unique(np.asarray(list(classes)))))
			if not np.array_equal(expected, actual):
				raise ValueError("classes must match the classes given during the initial call")

		y_encoded = np.array([self._class_to_index[val] for val in y_arr], dtype=np.int64)

		self.n_features_in_ = X_arr.shape[1]
		self.n_layers_ = len(self.hidden_layer_sizes) + 2
		self.out_activation_ = "logistic" if self.n_outputs_ == 2 else "softmax"

		if self.model_ is None:
			self._initialize_model(self.n_features_in_, len(self.classes_))  # type: ignore[arg-type]
			self.t_ = 0

		X_tensor = torch.as_tensor(X_arr, dtype=torch.float32, device=self._device)
		y_tensor = torch.as_tensor(y_encoded, dtype=torch.long, device=self._device)
		self.batch_size_ = self._resolve_batch_size(X_tensor.shape[0])
		self._run_training_loop(X_tensor, y_tensor, None, max_epochs=1)
		self._update_coefs_and_intercepts()
		return self

	def get_params(self, deep: bool = True) -> Dict[str, Any]:
		return {
			"hidden_layer_sizes": self._hidden_layer_sizes_param,
			"activation": self.activation,
			"solver": self.solver,
			"alpha": self.alpha,
			"batch_size": self.batch_size,
			"learning_rate": self.learning_rate,
			"learning_rate_init": self.learning_rate_init,
			"power_t": self.power_t,
			"max_iter": self.max_iter,
			"shuffle": self.shuffle,
			"random_state": self.random_state,
			"tol": self.tol,
			"verbose": self.verbose,
			"warm_start": self.warm_start,
			"momentum": self.momentum,
			"nesterovs_momentum": self.nesterovs_momentum,
			"early_stopping": self.early_stopping,
			"validation_fraction": self.validation_fraction,
			"beta_1": self.beta_1,
			"beta_2": self.beta_2,
			"epsilon": self.epsilon,
			"n_iter_no_change": self.n_iter_no_change,
			"max_fun": self.max_fun,
		}

	def set_params(self, **params: Any) -> "MLPClassifier":
		for key, value in params.items():
			if not hasattr(self, key) and key != "hidden_layer_sizes":
				raise ValueError(f"Invalid parameter {key} for MLPClassifier.")
			if key == "hidden_layer_sizes":
				self._hidden_layer_sizes_param = value
				self.hidden_layer_sizes = self._validate_hidden_layer_sizes(value)
				continue
			setattr(self, key, value)
		return self

	def fit(self, X: Any, y: Any) -> "MLPClassifier":
		X_arr = _as_float_array(X)
		y_arr = _as_label_array(y)
		if X_arr.shape[0] != y_arr.shape[0]:
			raise ValueError("X and y must contain the same number of samples")

		self._rng = _create_random_state(self.random_state)
		y_encoded = self._prepare_target(y_arr)
		self.n_features_in_ = X_arr.shape[1]
		self.n_layers_ = len(self.hidden_layer_sizes) + 2
		self.n_outputs_ = len(self.classes_) if self.classes_ is not None else None
		self.out_activation_ = "logistic" if self.n_outputs_ == 2 else "softmax"

		if not self.warm_start or self.model_ is None:
			self._initialize_model(self.n_features_in_, len(self.classes_))  # type: ignore[arg-type]
			self.loss_curve_ = []
			self.t_ = 0
			self.n_iter_ = 0
		else:
			self._validate_warm_start(X_arr.shape[1], len(self.classes_))  # type: ignore[arg-type]

		self._fit_internal(X_arr, y_encoded)
		self._update_coefs_and_intercepts()
		return self
