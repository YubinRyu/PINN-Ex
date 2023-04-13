import torch
from utils import net_o

def predict(train, device, model, X_lb, X_ub, Y_lb, Y_ub):
    x = torch.tensor(train[:, 0:1], requires_grad=True).float().to(device)
    y = torch.tensor(train[:, 1:2], requires_grad=True).float().to(device)
    z = torch.tensor(train[:, 2:3], requires_grad=True).float().to(device)
    input_C = torch.tensor(train[:, 3:4], requires_grad=True).float().to(device)
    input_T = torch.tensor(train[:, 4:5], requires_grad=True).float().to(device)
    domain = torch.tensor(train[:, 5:6], requires_grad=True).float().to(device)

    model.eval()
    output = net_o(x, y, z, input_C, input_T, domain, model, X_lb, X_ub)

    Y_lb = Y_lb.detach().cpu().numpy()
    Y_ub = Y_ub.detach().cpu().numpy()
    output = output.detach().cpu().numpy()

    output = (Y_lb * (1.0 - output) + Y_ub * (1.0 + output)) / 2.0

    u = output[:, 0:1]
    v = output[:, 1:2]
    w = output[:, 2:3]
    p = output[:, 3:4]
    T = output[:, 4:5]
    I = output[:, 5:6]
    M = output[:, 6:7]

    return u, v, w, p, T, I, M
