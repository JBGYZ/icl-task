# import torch
# from torch.utils.data import IterableDataset, DataLoader
# import torch
# import torch.nn as nn
# import torch.distributions as dist
# import torch.nn.functional as F
# from typing import Callable, Tuple, Any, List
# import dataclasses
# from torch.utils.tensorboard import SummaryWriter
# from models import Ridge, MLP

# ########################################################################################################################
# # Noisy Linear Regression                                                                                              #
# ########################################################################################################################

# @dataclasses.dataclass
# class NoisyLinearRegression:
#     n_tasks: int
#     n_dims: int
#     n_points: int
#     batch_size: int
#     data_seed: int
#     task_seed: int
#     noise_seed: int
#     data_scale: float
#     task_scale: float
#     noise_scale: float
#     dtype: Any

#     def __post_init__(self):
#         self.data_key = torch.manual_seed(self.data_seed)
#         self.task_key = torch.manual_seed(self.task_seed)
#         self.noise_key = torch.manual_seed(self.noise_seed)
#         self.task_pool = self.generate_task_pool() if self.n_tasks > 0 else None

#     @property
#     def name(self) -> str:
#         return f"NoisyLinReg({self.n_tasks})"

#     @classmethod
#     def from_task_pool(cls, task_pool: torch.Tensor, **kwargs) -> "NoisyLinearRegression":
#         assert kwargs["n_tasks"] == task_pool.size(0)
#         task = cls(**kwargs)
#         task.task_pool = task_pool
#         return task

#     def generate_task_pool(self) -> torch.Tensor:
#         torch.manual_seed(self.task_seed)
#         shape = (self.n_tasks, self.n_dims, 1)
#         tasks = torch.normal(0, 1, size=shape, dtype=self.dtype) * self.task_scale
#         return tasks

#     def sample_data(self, step: int) -> torch.Tensor:
#         torch.manual_seed(self.data_seed + step)
#         shape = (self.batch_size, self.n_points, self.n_dims)
#         data = torch.normal(0, 1, size=shape, dtype=self.dtype) * self.data_scale
#         return data

#     def sample_tasks(self, step: int) -> torch.Tensor:
#         torch.manual_seed(self.task_seed + step)
#         if self.n_tasks > 0:
#             idxs = torch.randint(0, self.n_tasks, (self.batch_size,))
#             tasks = self.task_pool[idxs]
#         else:
#             shape = (self.batch_size, self.n_dims, 1)
#             tasks = torch.normal(0, 1, size=shape, dtype=self.dtype) * self.task_scale
#         return tasks

#     def evaluate(self, data: torch.Tensor, tasks: torch.Tensor, step: int) -> torch.Tensor:
#         targets = torch.matmul(data, tasks)[:, :, 0]
#         torch.manual_seed(self.noise_seed + step)
#         noise = torch.normal(0, 1, size=targets.shape, dtype=self.dtype) * self.noise_scale
#         return targets + noise

#     def sample_batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         data, tasks = self.sample_data(step), self.sample_tasks(step)
#         targets = self.evaluate(data, tasks, step)
#         data_true = torch.concat((data.view(256, -1), targets[:,:-1]), dim=-1)
#         targets_true = torch.matmul(data, tasks)[:, -1, 0]
#         return data_true, targets_true.unsqueeze(-1)

#     @staticmethod
#     def evaluate_oracle(data: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
#         targets = torch.matmul(data, tasks)[:, -1, 0]
#         return targets



# noisyLinearRegression = NoisyLinearRegression(
#     n_tasks=2**4,
#     n_dims=8,
#     n_points=16,
#     batch_size=256,
#     data_seed=0,
#     task_seed=0,
#     noise_seed=0,
#     data_scale=1.0,
#     task_scale=1.0,
#     noise_scale=0.25,
#     dtype=torch.float32,
# )

# noisyLinearRegression.__post_init__()



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print device
# print(device)
# # Initialize the model input dim needs to match the task
# model = MLP(input_dim=143, output_dim=1, n_hidden_layers=6, n_hidden_neurons=512)
# # print model parameter number
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# model.to(device)

# # Initialize the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# # Initialize the loss function
# loss_fn = nn.MSELoss()

# # Training loop
# max_iterations = 500000  # Define the maximum number of iterations
# writer = SummaryWriter(log_dir=f'realruns/smalltask')  # Initialize the tensorboard writer

# newtasks = NoisyLinearRegression(
#     n_tasks=2**4,
#     n_dims=8,
#     n_points=16,
#     batch_size=256,
#     data_seed=-max_iterations,
#     task_seed=-max_iterations,
#     noise_seed=-max_iterations,
#     data_scale=1.0,
#     task_scale=1.0,
#     noise_scale=0.25,
#     dtype=torch.float32,
# )

# newtasks.__post_init__()

# for step in range(max_iterations):
#     # Your training logic goes here
#     data_true, targets_true = noisyLinearRegression.sample_batch(step)
#     data_true, targets_true = data_true.to(device), targets_true.to(device)
#     optimizer.zero_grad()
#     outputs = model(data_true)
#     loss = loss_fn(outputs, targets_true)
#     loss.backward()
#     optimizer.step()
#     # print every 100 steps
#     if step % 100 == 0:
#         print(f"Step {step} | Train Loss {loss.item()/8}")
#         writer.add_scalar("Loss/train", loss.item()/8, step)
#         # evaluate on test set
#         model.eval()
#         data_true, targets_true = newtasks.sample_batch(step)
#         data_true, targets_true = data_true.to(device), targets_true.to(device)
#         outputs = model(data_true)
#         loss = loss_fn(outputs, targets_true)
#         print(f"Step {step} | Test Loss {loss.item()/8}")
#         writer.add_scalar("Loss/test", loss.item()/8, step)
#         model.train()

# # Continue with other post-training tasks


import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from models import MLP, TransformerEncoder
from data_generator import NoisyLinearRegression, NoisyLinearRegression_trf
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def parse_args():
    parser = argparse.ArgumentParser(description="Noisy Linear Regression Training Script")
    parser.add_argument("--n_tasks", type=int, default=2**4, help="Number of tasks")
    parser.add_argument("--n_dims", type=int, default=8, help="Number of dimensions")
    parser.add_argument("--n_points", type=int, default=16, help="Number of data points")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--data_seed", type=int, default=0, help="Data seed")
    parser.add_argument("--task_seed", type=int, default=0, help="Task seed")
    parser.add_argument("--noise_seed", type=int, default=0, help="Noise seed")
    parser.add_argument("--data_scale", type=float, default=1.0, help="Data scale")
    parser.add_argument("--task_scale", type=float, default=1.0, help="Task scale")
    parser.add_argument("--noise_scale", type=float, default=0.25, help="Noise scale")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type (e.g., float32)")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--log_dir", type=str, default="realruns/smalltask", help="Tensorboard log directory")
    parser.add_argument("--max_iterations", type=int, default=500000, help="Maximum number of training iterations")

    parser.add_argument("--network", type=str, default="mlp", help="Network type (e.g., mlp, transformer)")
    parser.add_argument("--input_dim", type=int, default=144, help="Input dimension")
    parser.add_argument("--output_dim", type=int, default=1, help="Output dimension")
    parser.add_argument("--n_hidden_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--n_hidden_neurons", type=int, default=512, help="Number of hidden neurons")
    parser.add_argument("--dim_feedforward", type=int, default=128, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")


    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  

    if args.network == "fcn":
        model = MLP(input_dim=(args.n_dims + 1) * args.n_points, 
                    output_dim=args.output_dim,
                    n_hidden_layers=args.n_hidden_layers,
                    n_hidden_neurons=args.n_hidden_neurons,)
        
        noisyLinearRegression = NoisyLinearRegression_trf(
        n_tasks=args.n_tasks,
        n_dims=args.n_dims,
        n_points=args.n_points,
        batch_size=args.batch_size,
        data_seed=args.data_seed,
        task_seed=args.task_seed,
        noise_seed=args.noise_seed,
        data_scale=args.data_scale,
        task_scale=args.task_scale,
        noise_scale=args.noise_scale,
        dtype=torch.float32,)
        noisyLinearRegression.__post_init__()

    elif args.network == "transformer":
        model = TransformerEncoder(
                        d_model=args.n_dims +1 ,
                        dim_feedforward=args.dim_feedforward,
                        scaleup_dim=args.n_hidden_neurons,
                        nhead=1,
                        num_layers=args.n_hidden_layers,
                        embedding_type="scaleup",
                        pos_encoder_type="learned",
                        dropout=args.dropout)
        
        noisyLinearRegression = NoisyLinearRegression_trf(
        n_tasks=args.n_tasks,
        n_dims=args.n_dims,
        n_points=args.n_points,
        batch_size=args.batch_size,
        data_seed=args.data_seed,
        task_seed=args.task_seed,
        noise_seed=args.noise_seed,
        data_scale=args.data_scale,
        task_scale=args.task_scale,
        noise_scale=args.noise_scale,
        dtype=torch.float32,)
        noisyLinearRegression.__post_init__()
    else:
        raise NotImplementedError


    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.MSELoss()

    # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10000, eta_min=0.0, last_epoch=-1)


    writer = SummaryWriter(log_dir=f"testtest/network_{args.network}_n_tasks_{args.n_tasks}_n_dims_{args.n_dims}__n_points_{args.n_points}_n_hidden_layers_{args.n_hidden_layers}_n_hidden_neurons_{args.n_hidden_neurons}_dim_feedforward_{args.dim_feedforward}_lr_{args.lr}")

    if args.network == "transformer":

        newtasks = NoisyLinearRegression_trf(
            n_tasks=args.n_tasks,
            n_dims=args.n_dims,
            n_points=args.n_points,
            batch_size=args.batch_size,
            data_seed=args.data_seed - args.max_iterations,
            task_seed=args.task_seed - args.max_iterations,
            noise_seed=args.noise_seed - args.max_iterations,
            data_scale=args.data_scale,
            task_scale=args.task_scale,
            noise_scale=args.noise_scale,
            dtype=torch.float32,)
        newtasks.__post_init__()
    else:
        newtasks = NoisyLinearRegression_trf(
        n_tasks=args.n_tasks,
        n_dims=args.n_dims,
        n_points=args.n_points,
        batch_size=args.batch_size,
        data_seed=args.data_seed - args.max_iterations,
        task_seed=args.task_seed - args.max_iterations,
        noise_seed=args.noise_seed - args.max_iterations,
        data_scale=args.data_scale,
        task_scale=args.task_scale,
        noise_scale=args.noise_scale,
        dtype=torch.float32,)
        newtasks.__post_init__()

    for step in range(args.max_iterations):
        data, targets = noisyLinearRegression.sample_batch(step)
        loss_total = []
        for i in range(args.n_points):
            data_copy = data.clone()
            data_copy[:, i*2+1:, :] = 0
            
            targets_true = targets[:, i].unsqueeze(-1)
            data_true, targets_true = data_copy.to(device), targets_true.to(device)
            optimizer.zero_grad()
            outputs = model(data_true)
            loss = loss_fn(outputs, targets_true)
            loss.backward()
            optimizer.step()
            loss_total += [loss.item()/args.n_dims]
        if step % 100 == 0:
            print(f"Step {step} | Train Loss {sum(loss_total)/len(loss_total)}")
        if step % 500 == 0:
            for j in range(args.n_points):
                print(f"Loss/train_{j}", loss_total[j - args.n_points//2 +1])
        # lr_scheduler.step()
        if step % 500 == 0:
            # evaluate on test set
            model.eval()
            data, targets = newtasks.sample_batch(step)
            loss_total = []
            for i in range(args.n_points):
                data_copy = data.clone()
                data_copy[:, i*2+1:, :] = 0
                targets_true = targets[:, i].unsqueeze(-1)
                data_true, targets_true = data_copy.to(device), targets_true.to(device)
                optimizer.zero_grad()
                outputs = model(data_true)
                loss = loss_fn(outputs, targets_true)
                loss.backward()
                optimizer.step()
                loss_total += [loss.item()/args.n_dims]
            model.train()
            print(f"Step {step} | Test Loss {sum(loss_total)/len(loss_total)}")
            for j in range(args.n_points):
                print(f"Loss/test_{j}", loss_total[j - args.n_points//2 +1])

if __name__ == "__main__":
    main()

# grun python main.py --n_tasks 16 --n_dims 8 --n_points 16 --batch_size 256 --data_seed 0 --task_seed 0 --noise_seed 0 --data_scale 1.0 --task_scale 1.0 --noise_scale 0.25 --dtype float32 --lr 0.001 --log_dir realruns/transformersmall --max_iterations 500000 --network transformer --input_dim 143 --output_dim 1 --n_hidden_layers 4 --n_hidden_neurons 128 --dim_feedforward 128 --dropout 0.1