import objax
import objax.functional as F
import objax.nn as nn
import numpy as np
import pandas as pd
import jax
jax.config.update('jax_platform_name', 'cpu')
# import jaxlib
import jax.numpy as jnp
import os
os.environ["XLA_GPU_DEVICE_ORDINAL"] = "gpu:5"


class NeuralNetwork(objax.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def __call__(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(inputs, labels, input_dim, hidden_dim, output_dim, lr, epoch, batch_size):
    # Create the model
    model = NeuralNetwork(input_dim, hidden_dim, output_dim)

    # Define the loss function
    @objax.Function.with_vars(model.vars())
    def loss_fn(x, y):
        return ((model(x) - y) ** 2).mean()

    # Define the optimizer
    optimizer = objax.optimizer.Adam(model.vars())

    num_batches = len(inputs) // batch_size

    grad_values = objax.GradValues(loss_fn, model.vars())

    for e in range(epoch):
        # Training loop
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x = inputs[start_idx:end_idx]
            y = labels[start_idx:end_idx]

            g, v = grad_values(x, y)  # Compute gradients and loss
            optimizer(lr, g)

        # Print the loss every 100 iterations
        if (e + 1) % 100 == 0:
            print(f"Iteration {e + 1}/{epoch}, Loss: {v}")

    return model


if __name__ == "__main__":
    # Load the data
    stellar_model_dir = "/nfsdata/share/stellarmodel/"
    stellar_model_name = "PARSEC_logage9p5_MH_n1p5_0p6.csv"
    data = pd.read_csv(stellar_model_dir + stellar_model_name)

    inputs = jnp.array(data[['Mini', 'MH']].values)
    labels = jnp.array(data['G_BPmag'].values).reshape(-1, 1)

    _gpu = jax.devices("gpu")[5]

    def gpu(a):
        return jax.device_put(a, _gpu)

    inputs, labels = gpu(inputs), gpu(labels)

    input_dim = 2
    hidden_dim = 5
    output_dim = 1

    lr = 1e-3
    epoch = 1000
    batch_size = 512

    # Train the model
    # trained_model = train_model(
    #     inputs, labels, 
    #     input_dim, hidden_dim, output_dim, 
    #     lr, epoch, batch_size
    #     )

    # # Save the trained model
    # save_path = stellar_model_dir + 'mass2Bp.npz'
    # objax.io.save_var_collection(save_path, trained_model.vars())