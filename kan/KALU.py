import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from .KANLayer import KANLayer
from .KAN import KAN
from .Symbolic_KANLayer import Symbolic_KANLayer

from tqdm import tqdm

class NALULayer(nn.Module):
    def __init__(self, dim, eps=1e-7, omega=20, device='cpu'):
        super().__init__()
        self.W_m = nn.Parameter(torch.normal(torch.zeros(dim, dim), torch.ones(dim, dim) / (dim**2)))
        self.M_m = nn.Parameter(torch.normal(torch.zeros(dim, dim), torch.ones(dim, dim) / (dim**2)))
        self.G = nn.Parameter(torch.zeros(dim))

        self.identity = torch.eye(dim)
        self.inverse_identity = torch.ones(dim) - self.identity
        
        self.eps = torch.tensor([eps])
        self.omega = torch.tensor([omega])
    
    def get_weight_mul(self):
        return self.identity + self.inverse_identity * torch.tanh(self.W_m) * torch.sigmoid(self.M_m)

    def get_gate(self):
        return torch.tanh(self.G)
    
    def forward(self, x):
        # Multiplication part of NALU
        weight_mul = self.get_weight_mul()
        log_space = torch.log(torch.max(torch.abs(x), self.eps.expand_as(x))) @ weight_mul
        mul_term = torch.exp(torch.min(log_space, self.omega.expand_as(log_space)))
        # fixes sign errors
        shape = list(x.shape) + [weight_mul.shape[-1]]
        msm1 = torch.mul(torch.sign(x).unsqueeze(-1).expand(shape), torch.abs(weight_mul))
        msm2 = 1 - torch.abs(weight_mul)
        msm = msm1 + msm2
        msv = msm.prod(axis = 1)
        mul_term = torch.mul(mul_term, msv)

        g = self.get_gate()
        return x + torch.mul(mul_term, g)

class KALULayer(KANLayer):
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, base_fun=torch.nn.SiLU(), eps=1e-7, omega=20, device='cpu'):
        super().__init__(in_dim=in_dim, out_dim=out_dim, num=num, k=k, base_fun=base_fun, device=device)
        self.nalu = NALULayer(dim=in_dim, eps=eps, omega=omega, device=device)

    def forward(self, x):
        nalu_output = self.nalu(x)
        return super().forward(nalu_output)

    def update_grid_from_samples(self, x):
        nalu_output = self.nalu(x)
        super().update_grid_from_samples(nalu_output)

class KALU(KAN):
    def __init__(self, width, grid=3, k=3, base_fun=torch.nn.SiLU(), symbolic_enabled=True, eps=1e-7, omega=20, device='cpu'):
        super(KAN, self).__init__()

        ### Initializing the numerical front ###

        self.act_fun = []
        self.weights = []
        self.biases = []
        self.depth = len(width) - 1
        self.width = width

        for l in range(self.depth):
            layer = KALULayer(in_dim=width[l], out_dim=width[l+1], num=grid, k=k, base_fun=base_fun, eps=eps, omega=omega, device=device)
            self.act_fun.append(layer)
            self.weights.append(layer.nalu.W_m)
            self.weights.append(layer.nalu.M_m)
            self.weights.append(layer.nalu.G)

            bias = nn.Linear(width[l+1], 1, bias=False, device=device)
            bias.weight.data *= 0
            self.biases.append(bias)

        self.biases = nn.ModuleList(self.biases)
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun
        
        ### Initializing the symbolic front ###
        self.symbolic_fun = []
        for l in range(self.depth):
            symbolic_layer = Symbolic_KALULayer(in_dim=width[l], out_dim=width[l + 1], device=device)
            self.symbolic_fun.append(symbolic_layer)
        
        self.symbolic_fun = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled
        
        self.device = device

    def update_grid_from_samples(self, x):
        for l in range(self.depth):
            self.forward(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])

    def forward(self, x):
        self.acts = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_std = []

        self.acts.append(x)
        for l in range(self.depth):
            x_numerical, preacts, postacts, postspline = self.act_fun[l](x)

            if self.symbolic_enabled == True:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x)
            else:
                x_symbolic = 0
                postacts_symbolic = 0
            
            x = x_numerical + x_symbolic
            postacts = postacts_numerical + postacts_symbolic

            grid_reshape = self.act_fun[l].grid.reshape(self.width[l+1], self.width[l], -1)
            input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
            output_range = torch.mean(torch.abs(postacts), dim=0)
            self.acts_scale.append(output_range / input_range)
            self.acts_scale_std.append(torch.std(postacts, dim=0))
            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())         

            x = x + self.biases[l].weight
            self.acts.append(x)
        
        return x

    def train(self, dataset, steps=50, lr=0.001, lamb=0.1, batch=-1):
        X_train, X_test, y_train, y_test = dataset["train_input"], dataset["test_input"], dataset["train_label"], dataset["test_label"]

        # Helper function for batch training
        def get_batch_mask(X_train, batch_size):
            if batch_size < 0:
                yield np.arange(0,len(X_train))
            else:
                indices = np.random.permutation(np.arange(0,len(X_train)))
                for i in range(len(indices) // batch_size):
                    yield indices[i * batch_size : (i + 1) * batch_size]

        # Helper function for KAN regularization
        def reg(acts_scale):
            lamb_l1 = 1
            lamb_entropy = 1
            lamb_coef = 1
            lamb_coefdiff = 1
            small_mag_threshold = 1e-16
            small_reg_factor = 1.0
            t = 20
            
            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)
            
            reg_ = 0.0
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )
                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy
            
            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1
            
            # regularize NALU weights
            for w in self.weights:
                L_reg = torch.max(torch.min(-w, w) + t)
                if L_reg > 0:
                    reg_ += L_reg
            
            return reg_

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0)
        mse = torch.nn.MSELoss()

        progress_bar = tqdm(range(steps))
        for epoch in progress_bar:
            for mask in get_batch_mask(X_train, batch):
                pred = self(X_train[mask])
                loss = mse(pred, y_train[mask])
                loss += lamb * reg(self.acts_scale)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred_train = self(X_train)
                loss_train = mse(pred_train, y_train)
                reg_train = loss_train.detach().item() * lamb * reg(self.acts_scale)

                pred_test = self(X_test)
                loss_test = mse(pred_test, y_test)
                progress_bar.set_description(f"Train MSE : {loss_train:.3} | Test MSE : {loss_test:.3} | Regularization : {reg_train:.3}")
                self.update_grid_from_samples(X_train)


    def plot(self, folder="./figures", beta=3, mask=False, mode="supervised", scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None):
        if not os.path.exists(folder):
            os.makedirs(folder)
        # matplotlib.use('Agg')
        depth = len(self.width) - 1
        for l in range(depth):
            w_large = 2.0
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    rank = torch.argsort(self.acts[l][:, i])
                    fig, ax = plt.subplots(figsize=(w_large, w_large))

                    num = rank.shape[0]

                    mask = self.act_fun[l].mask.reshape(self.width[l + 1], self.width[l])[j][i]
                    if mask > 0.:
                        color = "black"
                        alpha_mask = 1
                    if mask == 0.:
                        color = "white"
                        alpha_mask = 0

                    if tick == True:
                        ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                        ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                        x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    if alpha_mask == 1:
                        plt.gca().patch.set_edgecolor('black')
                    else:
                        plt.gca().patch.set_edgecolor('white')
                    plt.gca().patch.set_linewidth(1.5)
                    # plt.axis('off')

                    plt.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                    if sample == True:
                        plt.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale ** 2)
                    plt.gca().spines[:].set_color(color)

                    lock_id = self.act_fun[l].lock_id[j * self.width[l] + i].long().item()
                    if lock_id > 0:
                        # im = plt.imread(f'{folder}/lock.png')
                        im = plt.imread(f'{RESOURCE_DIR}/lock.png')
                        newax = fig.add_axes([0.15, 0.7, 0.15, 0.15])
                        plt.text(500, 400, lock_id, fontsize=15)
                        newax.imshow(im)
                        newax.axis('off')

                    plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=400)
                    plt.close()