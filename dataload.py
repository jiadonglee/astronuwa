import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class CMDdata(Dataset):

    def __init__(self, fname, device=torch.device('cpu'), mode='delta', n_star=10000):
        self.device = device
        # data_dir = "/nfsdata/users/jdli_ny/wlkernel/mock/"
        n_train = 5000
        self.n_star = n_star
        self.mode = mode
        # fname = data_dir + f'cmd_moh_m0p5_0_bg_{n_train}tr_{self.n_star}.npz'
        data = np.load(fname, allow_pickle=True)

        self.bprp = data['bprp']
        mg = data['mg']
        theta = data['theta']

        theta_reshaped = np.expand_dims(theta, axis=1)
        theta_tiled    = np.tile(theta_reshaped, (1, self.n_star, 1))
        self.theta_mg = np.concatenate((mg.reshape(-1, self.n_star, 1), theta_tiled), axis=-1)
        # self.theta_mg = theta_mg
        # self.bprp = bprp
        if mode=='delta':
            self.delta_bprp = data['delta_bprp']

    def __len__(self):
        return len(self.theta_mg)

    def __getitem__(self, index):

        theta_mg_sample = self.theta_mg[index]
        theta_mg_torch = torch.from_numpy(theta_mg_sample).to(self.device, torch.float32)

        if self.mode=='delta':

            delta_bprp_sample = self.delta_bprp[index]
            y = torch.from_numpy(delta_bprp_sample.reshape(self.n_star, 1)).to(self.device, torch.float32)

        else:
            bprp_sample = self.bprp[index]
            y = torch.from_numpy(bprp_sample.reshape(self.n_star, 1)).to(self.device, torch.float32)

        return theta_mg_torch, y
    

    
    
if __name__=="__main__":    

    # Load data

    # Convert to PyTorch tensors
    # theta_mg_torch = torch.from_numpy(theta_mg).to(torch.float32)
    # bprp_torch = torch.from_numpy(bprp.reshape(-1, n_star, 1)).to(torch.float32)

    # Create dataset
    n_train = 5000
    n_star  = 10000
    fname = f'cmd_moh_m0p5_0_abg_{n_train}tr_{n_star}.npz'
    dataset = CMDdata("/nfsdata/users/jdli_ny/wlkernel/mock/"+fname)


    n_train = 5000
    # Split dataset into training and test sets
    n_test = int(0.2 * n_train)
    train_dataset, test_dataset = random_split(dataset, [n_train - n_test, n_test])

    # Create data loaders
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=True)