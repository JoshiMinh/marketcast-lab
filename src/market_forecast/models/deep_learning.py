from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ForecastModel
from .machine_learning import lag_matrix


class SequenceNetwork(nn.Module):
    def __init__(self, kind: str, hidden_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        recurrent = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.recurrent = recurrent(
            1, hidden_size, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0, batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.recurrent(values)
        return self.output(encoded[:, -1, :])


class TorchSequenceModel(ForecastModel):
    def __init__(
        self, *, kind: str, lookback: int = 20, hidden_size: int = 8, layers: int = 1,
        dropout: float = 0.0, epochs: int = 3, batch_size: int = 32,
        learning_rate: float = 0.01, patience: int = 2, seed: int = 42, device: str = "cpu",
    ) -> None:
        self.kind, self.lookback, self.hidden_size = kind, lookback, hidden_size
        self.layers, self.dropout, self.epochs = layers, dropout, epochs
        self.batch_size, self.learning_rate, self.patience = batch_size, learning_rate, patience
        self.seed, self.device_name = seed, device

    def _seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)

    def fit(self, values: np.ndarray) -> "TorchSequenceModel":
        self._seed()
        history = np.asarray(values, dtype=np.float32).reshape(-1)
        self.mean_, self.scale_ = float(history.mean()), float(history.std() or 1.0)
        scaled = (history - self.mean_) / self.scale_
        features, target = lag_matrix(scaled, self.lookback)
        tensors = TensorDataset(
            torch.tensor(features[:, :, None], dtype=torch.float32),
            torch.tensor(target[:, None], dtype=torch.float32),
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(tensors, batch_size=self.batch_size, shuffle=True, generator=generator)
        self.device_ = torch.device(self.device_name)
        self.network_ = SequenceNetwork(self.kind, self.hidden_size, self.layers, self.dropout).to(self.device_)
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.learning_curve_ = []
        best_loss, stale, best_state = float("inf"), 0, None
        for _ in range(self.epochs):
            losses = []
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.network_(batch_x.to(self.device_)), batch_y.to(self.device_))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            epoch_loss = float(np.mean(losses))
            self.learning_curve_.append(epoch_loss)
            if epoch_loss < best_loss - 1e-8:
                best_loss, stale = epoch_loss, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.network_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.history_ = history.tolist()
        return self

    def predict(self, horizon: int, context: np.ndarray | None = None) -> np.ndarray:
        history = list(np.asarray(context, dtype=float).reshape(-1)) if context is not None else list(self.history_)
        self.network_.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(horizon):
                window = (np.asarray(history[-self.lookback:]) - self.mean_) / self.scale_
                tensor = torch.tensor(window[None, :, None], dtype=torch.float32, device=self.device_)
                value = float(self.network_(tensor).cpu().item() * self.scale_ + self.mean_)
                predictions.append(value)
                history.append(value)
        return np.asarray(predictions)

    def get_params(self) -> dict[str, object]:
        return {
            "kind": self.kind, "lookback": self.lookback, "hidden_size": self.hidden_size,
            "layers": self.layers, "dropout": self.dropout, "epochs": self.epochs,
            "batch_size": self.batch_size, "learning_rate": self.learning_rate,
            "patience": self.patience, "seed": self.seed, "device": self.device_name,
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            "learning_curve": self.learning_curve_,
            "parameter_count": sum(parameter.numel() for parameter in self.network_.parameters()),
        }

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.network_.state_dict(), "params": self.get_params()}, path)
