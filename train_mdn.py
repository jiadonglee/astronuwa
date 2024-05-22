import numpy as np
import vaex
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import math
from dataload import CMDdata


class MDN(nn.Module):
    def __init__(self, n_input, n_hidden, n_gaussians):
        super(MDN, self).__init__()

        self.z_h = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 2*n_hidden),
            nn.Tanh(),
            nn.Linear(2*n_hidden, 4*n_hidden),
            nn.Tanh(),
            nn.Linear(4*n_hidden, 2*n_hidden),
            nn.Tanh(),
            nn.Linear(2*n_hidden, n_hidden),
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)  

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, mu, sigma
    

def gaussian_distribution(y, mu, sigma):
    oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*math.pi) # normalization factor for Gaussians
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI

def mdn_loss_fn(y, pi, mu, sigma):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=-1)
    # result = -torch.log(result)
    result = -torch.log(torch.clamp(result, min=1e-6))
    result = torch.mean(result)
    return result


def train_model(model, optimizer, train_loader, test_loader, num_epochs):

    for epoch in range(num_epochs):

        model.train()  # Set the model to training mode
        start_time = time.time()
        epoch_time = 0

        for t, c in train_loader:

            batch_start_time = time.time()

            pi, mu, sigma = model(t)
            loss = mdn_loss_fn(c, pi, mu, sigma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start_time  # Calculate the elapsed time for the batch
            epoch_time += batch_time  # Accumulate the elapsed time for the epoch

        if epoch % 100 == 0:
            train_loss = 0
            test_model(model, test_loader, epoch)  # Perform validation
            train_loss += loss.item() # Accumulate the train loss

            epoch_time = time.time() - start_time  # Calculate the elapsed time for the epoch
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.2f}, Epoch Time: {epoch_time:.2f}s, Batch Time: {batch_time:.2f}s")

    return loss.item()


def test_model(model, test_loader, epoch):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        val_loss = 0

        for t, c in test_loader:
            pi, mu, sigma = model(t)
            loss = mdn_loss_fn(c, pi, mu, sigma)
            val_loss += loss.item() * len(t)

        val_loss /= len(test_loader.dataset)
        print(f"Epoch: {epoch}, Validation Loss: {val_loss:.2f}")



if __name__=="__main__":  

    device = torch.device(0)
    n_train = 5000
    n_star  = 10000
    # fname = f'cmd_moh_m0p5_0_abg_{n_train}tr_{n_star}.npz'
    # fname = f'cmd_moh_m0p2_0_abg_{n_train}tr_{n_star}_deltacolor.npz'
    # fname = f'cmd_moh_m0p2_0_bg_{n_train}tr_{n_star}_deltacolor.npz'
    # fname = f'cmd_moh_0_bg_{n_train}tr_{n_star}_deltacolor.npz'
    # fname = "cmd_moh_0_sigma0m2_bg_5000tr_10000.npz"
    # dataset = CMDdata("/nfsdata/users/jdli_ny/wlkernel/mock/"+fname, device=device, mode='delta')

    mh_edges = np.array([-1., -0.6, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.5])

    mh_mids = mh_edges[:-1] + 0.5*np.diff(mh_edges)
    mh_mids = [np.round(_, 2) for _ in mh_mids]


    for k, moh in enumerate(mh_mids):

        print(f"Training for [Fe/H] = {moh}")

        dataset = CMDdata(
            f"/nfsdata/users/jdli_ny/bf/train_1025_twin/cmd_moh_{moh}.npz", 
            device=device, 
            n_star=n_star,
            mode=None
            )
        
        # Split dataset into training and test sets
        n_test = int(0.2*n_train)
        train_dataset, test_dataset = random_split(dataset, [n_train-n_test, n_test])

        # Create data loaders
        batch_size = 512

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)
        
        model = MDN(n_input=3, n_hidden=16, n_gaussians=2).to(device)
        
        # model.load_state_dict(
        #     torch.load(f'/nfsdata/users/jdli_ny/bf/mdn_modl/dd/mdn_1018_h32_bg_{moh}.pt')
        #     )

        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-9)
        
        num_epochs = 5000

        final_loss = train_model(model, optimizer, train_loader, test_loader, num_epochs)

        torch.save(model.state_dict(), 
                   f'/nfsdata/users/jdli_ny/bf/mdn_modl/dd/mdn_1025_h16_bt_{moh}.pt')
        
    #     if k==0:
    #         loss_df = vaex.from_arrays(
    #             moh=[moh], loss=[final_loss]
    #             )
    #     else:
    #         loss_df = loss_df.concat(vaex.from_arrays(moh=[moh], loss=[final_loss]))

    # loss_df.export(f'temp/mdn_1019_h16_bg_loss.csv')
