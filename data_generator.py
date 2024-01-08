import torch
from torch.utils.data import IterableDataset, DataLoader
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from typing import Callable, Tuple, Any, List
import dataclasses

########################################################################################################################
# Noisy Linear Regression                                                                                              #
########################################################################################################################

@dataclasses.dataclass
class NoisyLinearRegression:
    n_tasks: int
    n_dims: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    dtype: Any

    def __post_init__(self):
        self.data_key = torch.manual_seed(self.data_seed)
        self.task_key = torch.manual_seed(self.task_seed)
        self.noise_key = torch.manual_seed(self.noise_seed)
        self.task_pool = self.generate_task_pool() if self.n_tasks > 0 else None

    @property
    def name(self) -> str:
        return f"NoisyLinReg({self.n_tasks})"

    @classmethod
    def from_task_pool(cls, task_pool: torch.Tensor, **kwargs) -> "NoisyLinearRegression":
        assert kwargs["n_tasks"] == task_pool.size(0)
        task = cls(**kwargs)
        task.task_pool = task_pool
        return task

    def generate_task_pool(self) -> torch.Tensor:
        torch.manual_seed(self.task_seed)
        shape = (self.n_tasks, self.n_dims, 1)
        tasks = torch.normal(0, 1, size=shape, dtype=self.dtype) * self.task_scale
        return tasks

    def sample_data(self, step: int) -> torch.Tensor:
        torch.manual_seed(self.data_seed + step)
        shape = (self.batch_size, self.n_points, self.n_dims)
        data = torch.normal(0, 1, size=shape, dtype=self.dtype) * self.data_scale
        return data

    def sample_tasks(self, step: int) -> torch.Tensor:
        torch.manual_seed(self.task_seed + step)
        if self.n_tasks > 0:
            idxs = torch.randint(0, self.n_tasks, (self.batch_size,))
            tasks = self.task_pool[idxs]
        else:
            shape = (self.batch_size, self.n_dims, 1)
            tasks = torch.normal(0, 1, size=shape, dtype=self.dtype) * self.task_scale
        return tasks

    def evaluate(self, data: torch.Tensor, tasks: torch.Tensor, step: int) -> torch.Tensor:
        targets = torch.matmul(data, tasks)[:, :, 0]
        torch.manual_seed(self.noise_seed + step)
        noise = torch.normal(0, 1, size=targets.shape, dtype=self.dtype) * self.noise_scale
        return targets + noise

    def sample_batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, tasks = self.sample_data(step), self.sample_tasks(step)
        targets = self.evaluate(data, tasks, step)

        # targets[:, -1] = 0
        data_true = torch.concat((data, targets.unsqueeze(-1)), dim=-1)

        # targets_true = torch.matmul(data, tasks)[:, -1, 0]

        # return data_true, targets_true.unsqueeze(-1)
        # return data_true, targets[:,-1].unsqueeze(-1)
        return data_true, targets

    @staticmethod
    def evaluate_oracle(data: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        targets = torch.matmul(data, tasks)[:, -1, 0]
        return targets
    
@dataclasses.dataclass
class NoisyLinearRegression_trf:
    n_tasks: int
    n_dims: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    dtype: Any

    def __post_init__(self):
        self.data_key = torch.manual_seed(self.data_seed)
        self.task_key = torch.manual_seed(self.task_seed)
        self.noise_key = torch.manual_seed(self.noise_seed)
        self.task_pool = self.generate_task_pool() if self.n_tasks > 0 else None

    @property
    def name(self) -> str:
        return f"NoisyLinReg({self.n_tasks})"

    @classmethod
    def from_task_pool(cls, task_pool: torch.Tensor, **kwargs) -> "NoisyLinearRegression":
        assert kwargs["n_tasks"] == task_pool.size(0)
        task = cls(**kwargs)
        task.task_pool = task_pool
        return task

    def generate_task_pool(self) -> torch.Tensor:
        torch.manual_seed(self.task_seed)
        shape = (self.n_tasks, self.n_dims, 1)
        tasks = torch.normal(0, 1, size=shape, dtype=self.dtype) * self.task_scale
        return tasks

    def sample_data(self, step: int) -> torch.Tensor:
        torch.manual_seed(self.data_seed + step)
        shape = (self.batch_size, self.n_points, self.n_dims)
        data = torch.normal(0, 1, size=shape, dtype=self.dtype) * self.data_scale
        return data

    def sample_tasks(self, step: int) -> torch.Tensor:
        torch.manual_seed(self.task_seed + step)
        if self.n_tasks > 0:
            idxs = torch.randint(0, self.n_tasks, (self.batch_size,))
            tasks = self.task_pool[idxs]
        else:
            shape = (self.batch_size, self.n_dims, 1)
            tasks = torch.normal(0, 1, size=shape, dtype=self.dtype) * self.task_scale
        return tasks

    def evaluate(self, data: torch.Tensor, tasks: torch.Tensor, step: int) -> torch.Tensor:
        targets = torch.matmul(data, tasks)[:, :, 0]
        torch.manual_seed(self.noise_seed + step)
        noise = torch.normal(0, 1, size=targets.shape, dtype=self.dtype) * self.noise_scale
        return targets + noise

    def sample_batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, tasks = self.sample_data(step), self.sample_tasks(step)
        targets = self.evaluate(data, tasks, step)

        data_true = torch.zeros([self.batch_size, self.n_points*2, self.n_dims+1])
        for i in range(self.n_points):
            data_true[:, i*2, :self.n_dims] = data[:, i, :]
            data_true[:, i*2+1, self.n_dims] = targets[:, i]
            # data_true[:, i*2+1, :] = targets[:, i].unsqueeze(-1)

        return data_true, targets

    @staticmethod
    def evaluate_oracle(data: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        targets = torch.matmul(data, tasks)[:, -1, 0]
        return targets
    
