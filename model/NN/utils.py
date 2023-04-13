import os
import math
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split

rho = 520
mu = 0.0072
k_i = 1.54 * (10 ** 14)
Ea = 15023

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh

        layer_list = list()

        # Neural Network
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))

        layerdict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerdict)

    def forward(self, x, lb, ub):
        z = 2.0 * (x - lb) / (ub - lb) - 1.0

        out = self.layers(z)

        return out

def parse_args():
    parser = argparse.ArgumentParser(description='NN Multi-GPU Training')

    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
    parser.add_argument('--start-epoch', default=1, type=int, help='Starting Epoch')
    parser.add_argument('--patience', default=200, type=int, help='for Early Stopping')
    parser.add_argument('--epoch', default=1e5, type=int, help='Number of Total Epochs')

    # GPU
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:3456', type=str)
    parser.add_argument('--distributed', type=int, default=True, help='--gpu ignored if True')

    # Layer
    parser.add_argument('--activation', type=str, default='torch.nn.Tanh')
    parser.add_argument('--layers', type=list, default=[6, 100,  100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 7])

    # Data
    parser.add_argument('--physics', type=int, default=4000, help='Number of Physics Data')
    parser.add_argument('--empirical', type=int, default=200000, help='Number of Empirical Data')

    args = parser.parse_args()
    return args

def data_load(empirical, physics, dir):

    data_total = pd.read_pickle(dir + '/data/total.pkl')
    data_total_ = data_total.sample(empirical, random_state=42)

    data_in = pd.read_pickle(dir + '/data/inner.pkl')
    data_in_ = data_in.sample(physics, random_state=42)

    data_out = pd.read_pickle(dir + '/data/outer.pkl')
    data_out_ = data_out.sample(physics, random_state=42)

    data = pd.concat([data_total_, data_in_, data_out_], axis=0, join='outer')

    # For Adaptive Sampling
    data_s = data_total.drop(data_total_.index)

    # Initiator inlet temperature = 360 K, Initiator inlet concentration = 0.00012 mol/L
    data_p = data_s[['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature', 'domain']].to_numpy()

    index = np.reshape(np.round(data_p[:, 4:5], 5) == 360, (data_p.shape[0],)) \
            * np.reshape(np.round(data_p[:, 3:4], 5) == 0.00012, (data_p.shape[0],))
    idx_12_360 = np.array([i for i, x in enumerate(index) if x == True])

    data_p = data_p[idx_12_360, :]

    del data_in, data_in_
    del data_out, data_out_
    del data_total, data_total_

    train = data[['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature', 'domain']].to_numpy()
    target = data[['x-velocity', 'y-velocity', 'z-velocity', 'Pressure', 'Temperature', 'Initiator', 'Monomer']].to_numpy()

    X_train, X_val, Y_train, Y_val = train_test_split(train, target, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5, random_state=42)

    return X_train, X_val, Y_train, Y_val, data_s, data_p

def net_o(x, y, z, input_C, input_T, domain, network, X_lb, X_ub):
    output = network(torch.cat([x, y, z, input_C, input_T, domain], dim=1).float(), X_lb.float(), X_ub.float())

    return output

def net_f_in(x, y, z, input_C, input_T, domain, network, X_lb, X_ub, Y_lb, Y_ub):
    output = net_o(x, y, z, input_C, input_T, domain, network, X_lb, X_ub)

    Y_lb = torch.tensor(Y_lb).float().to(torch.cuda.current_device())
    Y_ub = torch.tensor(Y_ub).float().to(torch.cuda.current_device())

    output = (Y_lb * (1.0 - output) + Y_ub * (1.0 + output)) / 2.0

    u = output[:, 0:1]
    v = output[:, 1:2]
    w = output[:, 2:3]
    p = output[:, 3:4]
    T = output[:, 4:5]
    I = output[:, 5:6]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    I_x = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]
    I_y = torch.autograd.grad(I, y, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]
    I_z = torch.autograd.grad(I, z, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), retain_graph=True, create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), retain_graph=True, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), retain_graph=True, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), retain_graph=True, create_graph=True)[0]

    vel = 10.471975499999981

    # Relative Velocity from MRF
    ur = u - vel * y
    vr = v + vel * x

    ur_x = u_x
    ur_y = u_y - vel
    ur_z = u_z

    vr_x = v_x + vel
    vr_y = v_y
    vr_z = v_z

    f_cont = u_x + v_y + w_z

    f_u = rho * (ur_x * ur + ur_y * vr + ur_z * w) + p_x - mu * (u_xx + u_yy + u_zz) - (vel * vel * x) - (2 * vel * vr)
    f_v = rho * (vr_x * ur + vr_y * vr + vr_z * w) + p_y - mu * (v_xx + v_yy + v_zz) - (vel * vel * y) + (2 * vel * ur)
    f_w = rho * (w_x * ur + w_y * vr + w_z * w) + p_z - mu * (w_xx + w_yy + w_zz)

    f_I = (u * I_x + u_x * I + v * I_y + v_y * I + w * I_z + w_z * I) - k_i * torch.exp(-Ea / torch.abs(T))

    return f_cont, f_u, f_v, f_w, f_I

def net_f_out(x, y, z, input_C, input_T, domain, network, X_lb, X_ub, Y_lb, Y_ub):
    output = net_o(x, y, z, input_C, input_T, domain, network, X_lb, X_ub)

    Y_lb = torch.tensor(Y_lb).float().to(torch.cuda.current_device())
    Y_ub = torch.tensor(Y_ub).float().to(torch.cuda.current_device())

    output = (Y_lb * (1.0 - output) + Y_ub * (1.0 + output)) / 2.0

    u = output[:, 0:1]
    v = output[:, 1:2]
    w = output[:, 2:3]
    p = output[:, 3:4]
    T = output[:, 4:5]
    I = output[:, 5:6]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    I_x = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]
    I_y = torch.autograd.grad(I, y, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]
    I_z = torch.autograd.grad(I, z, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), retain_graph=True, create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), retain_graph=True, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), retain_graph=True, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), retain_graph=True, create_graph=True)[0]

    f_cont = u_x + v_y + w_z

    f_u = rho * (u_x * u + u_y * v + u_z * w) + p_x - mu * (u_xx + u_yy + u_zz)
    f_v = rho * (v_x * u + v_y * v + v_z * w) + p_y - mu * (v_xx + v_yy + v_zz)
    f_w = rho * (w_x * u + w_y * v + w_z * w) + p_z - mu * (w_xx + w_yy + w_zz)

    f_I = (u * I_x + u_x * I + v * I_y + v_y * I + w * I_z + w_z * I) - k_i * torch.exp(-Ea / torch.abs(T))

    return f_cont, f_u, f_v, f_w, f_I

def train(optimizer, model, data_loader, device, X_lb, X_ub, Y_lb, Y_ub):
    model.train()

    running_train_loss = 0.0
    running_1_loss = 0.0
    running_2_loss = 0.0
    running_3_loss = 0.0
    running_4_loss = 0.0
    running_5_loss = 0.0
    running_6_loss = 0.0
    running_7_loss = 0.0
    running_8_loss = 0.0

    for train_batch, target_batch in data_loader:
        loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8 = train_loss(train_batch, target_batch, device, model, X_lb, X_ub, Y_lb, Y_ub)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        running_train_loss += loss
        running_1_loss += loss1
        running_2_loss += loss2
        running_3_loss += loss3
        running_4_loss += loss4
        running_5_loss += loss5
        running_6_loss += loss6
        running_7_loss += loss7
        running_8_loss += loss8

    length = len(data_loader)
    train_loss_value = running_train_loss / length
    loss1 = running_1_loss / length
    loss2 = running_2_loss / length
    loss3 = running_3_loss / length
    loss4 = running_4_loss / length
    loss5 = running_5_loss / length
    loss6 = running_6_loss / length
    loss7 = running_7_loss / length
    loss8 = running_8_loss / length

    return train_loss_value, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8

def train_loss(train_batch, target_batch, device, network, X_lb, X_ub, Y_lb, Y_ub):
    x = torch.tensor(train_batch[:, 0:1], requires_grad=True).float().to(device)
    y = torch.tensor(train_batch[:, 1:2], requires_grad=True).float().to(device)
    z = torch.tensor(train_batch[:, 2:3], requires_grad=True).float().to(device)
    input_C = torch.tensor(train_batch[:, 3:4], requires_grad=True).float().to(device)
    input_T = torch.tensor(train_batch[:, 4:5], requires_grad=True).float().to(device)
    domain = torch.tensor(train_batch[:, 5:6], requires_grad=True).float().to(device)

    u = torch.tensor(target_batch[:, 0:1], requires_grad=True).float().to(device)
    v = torch.tensor(target_batch[:, 1:2], requires_grad=True).float().to(device)
    w = torch.tensor(target_batch[:, 2:3], requires_grad=True).float().to(device)
    p = torch.tensor(target_batch[:, 3:4], requires_grad=True).float().to(device)
    T = torch.tensor(target_batch[:, 4:5], requires_grad=True).float().to(device)
    I = torch.tensor(target_batch[:, 5:6], requires_grad=True).float().to(device)
    M = torch.tensor(target_batch[:, 6:7], requires_grad=True).float().to(device)

    index = np.reshape(train_batch[:, 5:6] == 1, (train_batch.shape[0],))
    in_idx = np.array([i for i, x in enumerate(index) if x == True])

    index = np.reshape(train_batch[:, 5:6] == 0, (train_batch.shape[0],))
    out_idx = np.array([i for i, x in enumerate(index) if x == True])

    x_in = torch.tensor(train_batch[in_idx, 0:1], requires_grad=True).float().to(device)
    y_in = torch.tensor(train_batch[in_idx, 1:2], requires_grad=True).float().to(device)
    z_in = torch.tensor(train_batch[in_idx, 2:3], requires_grad=True).float().to(device)
    input_C_in = torch.tensor(train_batch[in_idx, 3:4], requires_grad=True).float().to(device)
    input_T_in = torch.tensor(train_batch[in_idx, 4:5], requires_grad=True).float().to(device)
    domain_in = torch.tensor(train_batch[in_idx, 5:6], requires_grad=True).float().to(device)

    x_out = torch.tensor(train_batch[out_idx, 0:1], requires_grad=True).float().to(device)
    y_out = torch.tensor(train_batch[out_idx, 1:2], requires_grad=True).float().to(device)
    z_out = torch.tensor(train_batch[out_idx, 2:3], requires_grad=True).float().to(device)
    input_C_out = torch.tensor(train_batch[out_idx, 3:4], requires_grad=True).float().to(device)
    input_T_out = torch.tensor(train_batch[out_idx, 4:5], requires_grad=True).float().to(device)
    domain_out = torch.tensor(train_batch[out_idx, 5:6], requires_grad=True).float().to(device)

    output = net_o(x, y, z, input_C, input_T, domain, network, X_lb, X_ub)
    f_cont_in, f_u_in, f_v_in, f_w_in, f_I_in = net_f_in(x_in, y_in, z_in, input_C_in, input_T_in, domain_in, network, X_lb, X_ub, Y_lb, Y_ub)
    f_cont_out, f_u_out, f_v_out, f_w_out, f_I_out = net_f_out(x_out, y_out, z_out, input_C_out, input_T_out, domain_out, network, X_lb, X_ub, Y_lb, Y_ub)


    loss1 = torch.mean(torch.abs(u - output[:, 0:1])) + torch.mean(torch.abs(v - output[:, 1:2])) + torch.mean(torch.abs(w - output[:, 2:3])) # UVW
    loss2 = torch.mean(torch.abs(p - output[:, 3:4])) # Pressure
    loss3 = torch.mean(torch.abs(T - output[:, 4:5])) # Temperature
    loss4 = torch.mean(torch.abs(I - output[:, 5:6])) # Initiator
    loss5 = torch.mean(torch.abs(M - output[:, 6:7])) # Monomer

    # Continuity
    loss6 = torch.mean(torch.abs(f_cont_in)) + torch.mean(torch.abs(f_cont_out))

    if math.isnan(loss6) == True:
        loss6 = torch.tensor(0).float().to(device)
    else:
        loss6 = loss6

    # Navier Stokes
    loss7 = torch.mean(torch.abs(f_u_in)) + torch.mean(torch.abs(f_v_in)) + torch.mean(torch.abs(f_w_in)) \
            + torch.mean(torch.abs(f_u_out)) + torch.mean(torch.abs(f_v_out)) + torch.mean(torch.abs(f_w_out))

    if math.isnan(loss7) == True:
        loss7 = torch.tensor(0).float().to(device)
    else:
        loss7 = loss7

    # Species Balance
    loss8 = torch.mean(torch.abs(f_I_in)) + torch.mean(torch.abs(f_I_out))

    if math.isnan(loss8) == True:
        loss8 = torch.tensor(0).float().to(device)
    else:
        loss8 = loss8

    loss = (loss1 + loss2 + loss3 + loss4 + loss5) * 100

    return loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8

def validation_loss(X_val, Y_val, device, network, X_lb, X_ub, Y_lb, Y_ub):

    x = torch.tensor(X_val[:, 0:1], requires_grad=True).float().to(device)
    y = torch.tensor(X_val[:, 1:2], requires_grad=True).float().to(device)
    z = torch.tensor(X_val[:, 2:3], requires_grad=True).float().to(device)
    input_C = torch.tensor(X_val[:, 3:4], requires_grad=True).float().to(device)
    input_T = torch.tensor(X_val[:, 4:5], requires_grad=True).float().to(device)
    domain = torch.tensor(X_val[:, 5:6], requires_grad=True).float().to(device)

    u = torch.tensor(Y_val[:, 0:1], requires_grad=True).float().to(device)
    v = torch.tensor(Y_val[:, 1:2], requires_grad=True).float().to(device)
    w = torch.tensor(Y_val[:, 2:3], requires_grad=True).float().to(device)
    p = torch.tensor(Y_val[:, 3:4], requires_grad=True).float().to(device)
    T = torch.tensor(Y_val[:, 4:5], requires_grad=True).float().to(device)
    I = torch.tensor(Y_val[:, 5:6], requires_grad=True).float().to(device)
    M = torch.tensor(Y_val[:, 6:7], requires_grad=True).float().to(device)

    index = np.reshape(X_val[:, 5:6] == 1, (X_val.shape[0],))
    in_idx = np.array([i for i, x in enumerate(index) if x == True])

    index = np.reshape(X_val[:, 5:6] == 0, (X_val.shape[0],))
    out_idx = np.array([i for i, x in enumerate(index) if x == True])

    x_in = torch.tensor(X_val[in_idx, 0:1], requires_grad=True).float().to(device)
    y_in = torch.tensor(X_val[in_idx, 1:2], requires_grad=True).float().to(device)
    z_in = torch.tensor(X_val[in_idx, 2:3], requires_grad=True).float().to(device)
    input_C_in = torch.tensor(X_val[in_idx, 3:4], requires_grad=True).float().to(device)
    input_T_in = torch.tensor(X_val[in_idx, 4:5], requires_grad=True).float().to(device)
    domain_in = torch.tensor(X_val[in_idx, 5:6], requires_grad=True).float().to(device)

    x_out = torch.tensor(X_val[out_idx, 0:1], requires_grad=True).float().to(device)
    y_out = torch.tensor(X_val[out_idx, 1:2], requires_grad=True).float().to(device)
    z_out = torch.tensor(X_val[out_idx, 2:3], requires_grad=True).float().to(device)
    input_C_out = torch.tensor(X_val[out_idx, 3:4], requires_grad=True).float().to(device)
    input_T_out = torch.tensor(X_val[out_idx, 4:5], requires_grad=True).float().to(device)
    domain_out = torch.tensor(X_val[out_idx, 5:6], requires_grad=True).float().to(device)

    output = net_o(x, y, z, input_C, input_T, domain, network, X_lb, X_ub)
    f_cont_in, f_u_in, f_v_in, f_w_in, f_I_in = net_f_in(x_in, y_in, z_in, input_C_in, input_T_in, domain_in, network, X_lb, X_ub, Y_lb, Y_ub)
    f_cont_out, f_u_out, f_v_out, f_w_out, f_I_out = net_f_out(x_out, y_out, z_out, input_C_out, input_T_out, domain_out, network, X_lb, X_ub, Y_lb, Y_ub)

    loss1 = torch.mean(torch.abs(u - output[:, 0:1])) + torch.mean(torch.abs(v - output[:, 1:2])) + torch.mean(torch.abs(w - output[:, 2:3]))  # UVW
    loss2 = torch.mean(torch.abs(p - output[:, 3:4]))  # Pressure
    loss3 = torch.mean(torch.abs(T - output[:, 4:5]))  # Temperature
    loss4 = torch.mean(torch.abs(I - output[:, 5:6]))  # Initiator
    loss5 = torch.mean(torch.abs(M - output[:, 6:7]))  # Monomer

    # Continuity
    loss6 = torch.mean(torch.abs(f_cont_in)) + torch.mean(torch.abs(f_cont_out))

    if math.isnan(loss6) == True:
        loss6 = torch.tensor(0).float().to(device)
    else:
        loss6 = loss6

    # Navier Stokes
    loss7 = torch.mean(torch.abs(f_u_in)) + torch.mean(torch.abs(f_v_in)) + torch.mean(torch.abs(f_w_in)) \
            + torch.mean(torch.abs(f_u_out)) + torch.mean(torch.abs(f_v_out)) + torch.mean(torch.abs(f_w_out))

    if math.isnan(loss7) == True:
        loss7 = torch.tensor(0).float().to(device)
    else:
        loss7 = loss7

    # Species Balance
    loss8 = torch.mean(torch.abs(f_I_in)) + torch.mean(torch.abs(f_I_out))

    if math.isnan(loss8) == True:
        loss8 = torch.tensor(0).float().to(device)
    else:
        loss8 = loss8

    val_loss = (loss1 + loss2 + loss3 + loss4 + loss5) * 100

    return val_loss

def adaptive_sampling(random, device, network, X_lb, X_ub, Y_lb, Y_ub):
    train_random = random[['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature', 'domain']].to_numpy()
    target_random = random[['x-velocity', 'y-velocity', 'z-velocity', 'Pressure', 'Temperature', 'Initiator', 'Monomer']].to_numpy()

    target_random = 2.0 * (target_random - Y_lb.numpy()) / (Y_ub.numpy() - Y_lb.numpy()) - 1.0

    train_random = torch.tensor(train_random, dtype=torch.float64)
    target_random = torch.tensor(target_random, dtype=torch.float64)

    train_random = torch.tensor(train_random, requires_grad=True).float().to(device)
    target_random = torch.tensor(target_random, requires_grad=True).float().to(device)

    output = net_o(train_random[:, 0:1], train_random[:, 1:2], train_random[:, 2:3], train_random[:, 3:4], train_random[:, 4:5], train_random[:, 5:6], network, X_lb, X_ub)

    residual = (torch.abs(target_random[:, 0:1] - output[:, 0:1]) + torch.abs(target_random[:, 1:2] - output[:, 1:2]) + torch.abs(target_random[:, 2:3] - output[:, 2:3])) / 3 \
               + torch.abs(target_random[:, 4:5] - output[:, 4:5]) + torch.abs(target_random[:, 5:6] - output[:, 5:6]) + torch.abs(target_random[:, 6:7] - output[:, 6:7])
    residual = residual.detach().cpu().numpy().ravel()

    index = np.argsort(-1.0 * np.abs(residual))[:200]

    points_X = train_random[index]
    points_Y = target_random[index]

    points_X = torch.tensor(points_X, dtype=torch.float64).detach().cpu()
    points_Y = torch.tensor(points_Y, dtype=torch.float64).detach().cpu()

    return points_X, points_Y

def plot(train, u_pred, v_pred, w_pred, points_X, points_Y, epoch, dir):
    points_x = points_X[:, 0:1]
    points_z = points_X[:, 2:3]

    nn = 100
    xx = np.linspace(-0.095, 0.095, nn)
    yy = np.linspace(0, 0, nn)
    zz = np.linspace(0.005, 0.295, nn)

    X, Y, Z = np.meshgrid(xx, yy, zz)

    u_predict = griddata(train[:, 0:3], u_pred[:].flatten(), (X, Y, Z), method='linear')
    v_predict = griddata(train[:, 0:3], v_pred[:].flatten(), (X, Y, Z), method='linear')
    w_predict = griddata(train[:, 0:3], w_pred[:].flatten(), (X, Y, Z), method='linear')

    # Speed = sqrt(u ** 2 + v ** 2 + w ** 2)
    levels = np.linspace(0, 0.6493596846124651, 1000)
    plt.contourf(X[0], Z[0], (u_predict[0] ** 2 + v_predict[0] ** 2 + w_predict[0] ** 2) ** 0.5, levels, cmap='jet')
    cb = plt.colorbar()
    plt.scatter(points_x, points_z, marker="x", s=12, c='red')
    plt.gca().invert_yaxis()
    plt.axis('off')

    plt.savefig(dir + '/result/NN/adaptive_sampling/contour_plot/{}.png'.format(epoch))

    plt.gca().invert_yaxis()
    cb.remove()

    PX = pd.DataFrame(np.array(points_X))
    PY = pd.DataFrame(np.array(points_Y))

    PX.to_csv(dir + '/result/NN/adaptive_sampling/points/X_{}.csv'.format(epoch), index=False, header=False)
    PY.to_csv(dir + '/result/NN/adaptive_sampling/points/Y_{}.csv'.format(epoch), index=False, header=False)
