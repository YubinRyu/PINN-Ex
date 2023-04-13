import sys
import time
import warnings

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *
from evaluate import predict

def main_worker(gpu, world_size, args):
    warnings.filterwarnings('ignore')

    # Directory Setting
    dir = os.path.dirname(os.path.abspath(__file__))

    pardir = os.path.dirname(dir)
    pardir = os.path.dirname(pardir)

    # Device Setting
    torch.cuda.set_device(gpu)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=world_size, rank=gpu)


    # Data Loading
    X_train, X_val, Y_train, Y_val, data_s, data_p = data_load(args.empirical, args.physics, pardir)

    if not args.distributed or (args.distributed and gpu == 0):
        print('Trainset Size : {}, Validation Size : {}'.format(X_train.shape[0], X_val.shape[0]))

    X_train = torch.tensor(X_train, dtype=torch.float64)
    Y_train = torch.tensor(Y_train, dtype=torch.float64)

    X_val = torch.tensor(X_val, dtype=torch.float64)
    Y_val = torch.tensor(Y_val, dtype=torch.float64)

    X_lb = torch.min(X_train, dim=0)[0]
    X_ub = torch.max(X_train, dim=0)[0]

    Y_lb = torch.min(Y_train, dim=0)[0]
    Y_ub = torch.max(Y_train, dim=0)[0]

    Y_train = 2.0 * (Y_train - Y_lb) / (Y_ub - Y_lb) - 1.0
    Y_val = 2.0 * (Y_val - Y_lb) / (Y_ub - Y_lb) - 1.0

    train_set = TensorDataset(X_train, Y_train)

    if args.distributed:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)


    # Model Setting
    model = DNN(args.layers)

    if gpu is not None:
        model.cuda(gpu)

    if args.distributed:
        ddp_model = DDP(model, device_ids=[gpu])
        model = ddp_model


    # Train Setting
    best_epoch = 1
    ti = time.time()
    trigger_times = 0
    patience = args.patience
    last_epoch = args.epoch
    best_val_loss = 100000.0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=False, min_lr=1e-6)


    # List for Loss Monitoring
    loss_list, val_loss_list = [], []
    uvw_loss_list, p_loss_list, t_loss_list, i_loss_list, m_loss_list = [], [], [], [], []
    cont_loss_list, NS_loss_list, spec_loss_list = [], [], []


    # Start Training
    if not args.distributed or (args.distributed and gpu == 0):
        sys.stdout.flush()

    for epoch in range(args.start_epoch, last_epoch+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8 = train(optimizer, model, train_loader, gpu, X_lb, X_ub, Y_lb, Y_ub)

        if epoch % 1 == 0:
            val_loss = validation_loss(X_val, Y_val, gpu, model, X_lb, X_ub, Y_lb, Y_ub)
            scheduler.step(val_loss)

            if not args.distributed or (args.distributed and gpu == 0):
                print('Iteration: %d, Loss: %.3e, Validation Loss: %.3e, Learning Rate: %.3e, Best Val Epoch: %d, Time: %.4f'
                      % (epoch, loss.item(), val_loss.item(), optimizer.param_groups[0]['lr'], best_epoch, time.time() - ti))
                print('uvw: %.3e, P: %.3e, T: %.3e, I: %.3e, M: %.3e, cont: %.3e, momentum: %.3e, species: %.3e'
                      % (loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item(), loss7.item(), loss8.item()))

            loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())

            uvw_loss_list.append(loss1.item())
            p_loss_list.append(loss2.item())
            t_loss_list.append(loss3.item())
            i_loss_list.append(loss4.item())
            m_loss_list.append(loss5.item())

            cont_loss_list.append(loss6.item())
            NS_loss_list.append(loss7.item())
            spec_loss_list.append(loss8.item())


            # Saving Best Model
            if torch.cuda.current_device() == 0:

                if val_loss.item() < best_val_loss:

                    best_val_loss = val_loss.item()
                    best_epoch = epoch

                    print('Best Validation Loss : ', best_val_loss)
                    print('Saving Best Model...')

                    CHECKPOINT_PATH = pardir + '/result/NN/' + 'best_model.pth'
                    torch.save(model.state_dict(), CHECKPOINT_PATH)

                    print('Trigger Times: 0')
                    trigger_times = 0

                    # Save checkpoint
                    torch.save({'epoch': epoch,
                                'loss': loss, 'val_loss': val_loss,
                                'best_epoch': best_epoch, 'trigger_times': trigger_times,
                                'loss_history': loss_list, 'val_loss_history': val_loss_list,
                                'uvw_loss_history': uvw_loss_list, 'P_loss_history': p_loss_list,
                                'T_loss_history': t_loss_list, 'I_loss_history': i_loss_list,
                                'M_loss_history': m_loss_list, 'cont_loss_history': cont_loss_list,
                                'NS_loss_history': NS_loss_list, 'spec_loss_history': spec_loss_list},
                                pardir + '/result/NN/' + 'checkpoint.pt')

                else:
                    trigger_times += 1
                    print('Trigger Times:', trigger_times)

                    if trigger_times >= patience:
                        print('Early Stopping!')
                        break

        if epoch % 1000 == 0:
            if not args.distributed or (args.distributed and gpu == 0):

                # 1. Randomly selecting 20,000 data points
                random = data_s.sample(20000)
                random = random[['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature', 'domain',
                                 'x-velocity', 'y-velocity', 'z-velocity', 'Pressure', 'Temperature', 'Initiator', 'Monomer']]


                # 2. Removing duplicates from current dataset
                current_dataset = (torch.cat((X_train, Y_train), dim=1)).numpy()

                current_dataset = pd.DataFrame(current_dataset)
                current_dataset.columns = ['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature', 'domain',
                                           'x-velocity', 'y-velocity', 'z-velocity', 'Pressure', 'Temperature', 'Initiator', 'Monomer']

                concat = pd.concat([random, current_dataset, current_dataset], axis=0, join='outer')
                train_s = concat.drop_duplicates(['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature'], keep=False)


                # 3. Adaptive sampling
                points_X, points_Y = adaptive_sampling(train_s, gpu, model, X_lb, X_ub, Y_lb, Y_ub)


                # 4. Updating dataset
                X_train = torch.cat((X_train, points_X), dim=0)
                Y_train = torch.cat((Y_train, points_Y), dim=0)


                # 5. Contour plot
                u_pred, v_pred, w_pred, p_pred, T_pred, I_pred, M_pred = predict(data_p, gpu, model, X_lb, X_ub, Y_lb, Y_ub)
                plot(data_p, u_pred, v_pred, w_pred, points_X, points_Y, epoch, pardir)

def main():
    args = parse_args()
    print(args)

    n_gpus = torch.cuda.device_count()
    print('number of gpus:', n_gpus)

    if args.distributed:
        world_size = torch.cuda.device_count() if n_gpus is None else n_gpus
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

    else:
        gpu = None
        main_worker(gpu, None, args)

if __name__ == "__main__":
    main()
