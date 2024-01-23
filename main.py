import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from models import MLP, TransformerEncoder, StackedRNN, MLPMixer, CNN, TransformerEncoderFCN, MLPMixerFCN
from data_generator import NoisyLinearRegression_trf
import argparse

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
        model = MLP(input_dim=(args.n_dims + 1) * args.n_points*2, 
                    output_dim=args.output_dim,
                    n_hidden_layers=args.n_hidden_layers,
                    n_hidden_neurons=args.n_hidden_neurons,)
    elif args.network == "transformer":
        model = TransformerEncoder(
                        d_model=args.n_dims +1 ,
                        dim_feedforward=args.dim_feedforward,
                        scaleup_dim=args.n_hidden_neurons,
                        nhead=2,
                        num_layers=args.n_hidden_layers,
                        embedding_type="scaleup",
                        pos_encoder_type="learned",
                        dropout=args.dropout)
    elif args.network == "transformer_fcn":
        model = TransformerEncoderFCN(
                        d_model=args.n_dims +1 ,
                        dim_feedforward=args.dim_feedforward,
                        scaleup_dim=args.n_hidden_neurons,
                        nhead=2,
                        num_layers=args.n_hidden_layers,
                        embedding_type="scaleup",
                        pos_encoder_type="learned",
                        dropout=args.dropout)
    elif args.network == "rnn":
        model = StackedRNN(input_size=args.n_dims + 1,
                        hidden_size=args.n_hidden_neurons,
                        num_layers=args.n_hidden_layers,
                        output_size=args.output_dim)
    elif args.network == "mlpmixer":
        model = MLPMixer(input_dim=args.n_dims + 1,
                        n_seq=args.n_points * 2,
                        n_hidden_neurons=args.n_hidden_neurons,
                        output_dim=args.output_dim,
                        n_hidden_layers=args.n_hidden_layers,
                        dropout=args.dropout)
    elif args.network == "mlpmixer_fcn":
        model = MLPMixerFCN(input_dim=args.n_dims + 1,
                        n_seq=args.n_points * 2,
                        n_hidden_neurons=args.n_hidden_neurons,
                        output_dim=args.output_dim,
                        n_hidden_layers=args.n_hidden_layers,
                        dropout=args.dropout)
    elif args.network == "cnn":
        model = CNN(embed_dim=args.n_points * args.dim_feedforward,
                    num_filters=args.dim_feedforward,
                    n_hidden_neurons=args.n_hidden_neurons,
                    n_hidden_layers=args.n_hidden_layers,
                    n_dim=args.n_dims,
                    dropout=args.dropout)
    else:
        raise NotImplementedError

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    writer = SummaryWriter(log_dir=f"testtest/network_{args.network}_n_tasks_{args.n_tasks}_n_dims_{args.n_dims}_n_points_{args.n_points}_n_hidden_layers_{args.n_hidden_layers}_n_hidden_neurons_{args.n_hidden_neurons}_dim_feedforward_{args.dim_feedforward}_lr_{args.lr}_batch_size_{args.batch_size}_seed_{args.data_seed}_max_iterations_{args.max_iterations}")

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

    for step in range(args.max_iterations):
        data, targets = noisyLinearRegression.sample_batch(step)
        loss_total = []
        i = args.n_points - 1
        data[:, i*2+1:, :] = 0
        targets_true = targets[:, i].unsqueeze(-1)
        data_true, targets_true = data.to(device), targets_true.to(device)
        optimizer.zero_grad()
        outputs = model(data_true)
        loss = loss_fn(outputs, targets_true)
        loss.backward()
        optimizer.step()
        loss_total += [loss.item()/args.n_dims]

        if step % 100 == 0:
            print(f"Step {step} | Train Loss {sum(loss_total)/len(loss_total)}")
            writer.add_scalar("Loss/train", sum(loss_total)/len(loss_total), step)

if __name__ == "__main__":
    main()