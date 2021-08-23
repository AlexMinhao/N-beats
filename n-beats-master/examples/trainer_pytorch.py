import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from data import get_m4_data, dummy_data_generator
from torch import optim
from torch.nn import functional as F

from nbeats_pytorch.model import NBeatsNet
import os
import time
from metrics import metric

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CHECKPOINT_NAME = 'nbeats-training-checkpoint.th'

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 1:
        lr_adjust = {epoch: args.lr * (0.95 ** ((epoch - 1) // 1))}
    elif args.lradj == 2:

        lr_adjust = {
            20: 0.0005, 40: 0.0001, 60: 0.00005, 80: 0.00001

        }
    elif args.lradj == 3:
        lr_adjust = {
            20: 0.0005, 25: 0.0001, 35: 0.00005, 55: 0.00001
            , 70: 0.000001
        }
    elif args.lradj == 4:
        lr_adjust = {
            30: 0.0005, 40: 0.0003, 50: 0.0001, 65: 0.00001
            , 80: 0.000001
        }
    elif args.lradj == 5:
        lr_adjust = {
            40: 0.0001, 60: 0.00005
        }
    elif args.lradj == 6:

        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10: 0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj == 61:
        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10: 0.001, 25: 0.0005, 35: 0.0001, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj == 7:
        lr_adjust = {
            10: 0.0001, 30: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj == 8:
        lr_adjust = {
            0: 0.0005, 5: 0.0008, 10: 0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj == 9:
        lr_adjust = {
            0: 0.0001, 10: 0.0005, 20: 0.001, 40: 0.0001, 45: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))




def get_script_arguments():
    parser = ArgumentParser(description='N-Beats')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--disable-plot', action='store_true', help='Disable interactive plots')
    parser.add_argument('--task', default='m4', choices=['m4', 'dummy'], required=False)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--lradj', type=int, default=6, help='adjust learning rate')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break


def main():
    args = get_script_arguments()
    device = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')
    forecast_length = 4
    backcast_length = 2 * forecast_length
    batch_size = 4  # greater than 4 for viz

    if args.task == 'm4':
        data_gen_train = batcher(get_m4_data(backcast_length, forecast_length, is_training= True), batch_size=batch_size, infinite=True)
        data_gen_test = batcher(get_m4_data(backcast_length, forecast_length, is_training= False), batch_size=batch_size,
                                 infinite=True)

    elif args.task == 'dummy':
        data_gen = dummy_data_generator(backcast_length, forecast_length,
                                        signal_type='seasonality', random=True,
                                        batch_size=batch_size)
    else:
        raise Exception('Unknown task.')

    print('--- Model ---')
    net = NBeatsNet(device=device,
                    stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[2, 8, 3],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=1024,
                    share_weights_in_stack=False,
                    nb_harmonics=None)

    optimiser = optim.Adam(net.parameters())

    def plot_model(x, target, grad_step):
        if not args.disable_plot:
            print('plot()')
            plot(net, x, target, backcast_length, forecast_length, grad_step)

    max_grad_steps = 10000
    if args.test:
        max_grad_steps = 5

    simple_fit(args, net, optimiser, data_gen_train, data_gen_test, plot_model, device, max_grad_steps)


def simple_fit(args, net, optimiser, data_generator_train, data_generator_test, on_save_callback, device, max_grad_steps=10000):
    print('--- Training ---')
    # initial_grad_step = load(net, optimiser)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimiser, epoch, args)
        epoch_start_time = time.time()
        net.train()
        loss_total = 0
        cnt = 0
        for grad_step, (x, target) in enumerate(data_generator_train):
                 # (4, 50)  (4, 10)
            # grad_step += initial_grad_step
            optimiser.zero_grad()
            net.zero_grad()
            Iter_start_time = time.time()
            backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
            loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
            loss.backward()
            optimiser.step()
            # print(f'grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}')
            loss_total += float(loss)
            cnt += 1
            if grad_step % 1000 == 0:
                print(
                 '| Iter {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(
                     epoch, (
                             time.time() - Iter_start_time), loss))
            # if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            #     with torch.no_grad():
            #         save(net, optimiser, grad_step)
            #         if on_save_callback is not None:
            #             on_save_callback(x, target, grad_step)
            # if grad_step > max_grad_steps:
            #     print('Finished.')
            #     break
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(
                epoch, (
                        time.time() - epoch_start_time), loss_total / cnt))

        validate(net,epoch,data_generator_test, device)



def validate(model, epoch, dataloader, device):
    start = time.time()
    print("===================Validate Normal=========================")
    preds = []
    trues = []
    model.eval()
    for grad_step, (x, target) in enumerate(dataloader):
        # (4, 50)  (4, 10)

        backcast, forecast = model(torch.tensor(x, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))

        preds.append(forecast.detach().cpu().numpy())
        trues.append(target.detach().cpu().numpy())

    mae, mse, rmse, mape, mspe, corr, mase, smape = metric(preds, trues)
    print(
        'Epoch:{} mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(epoch,
            mse, mae, rmse, mape, mspe, corr, mase, smape))


def save(model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, CHECKPOINT_NAME)


def load(model, optimiser):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0


def plot(net, x, target, backcast_length, forecast_length, grad_step):
    net.eval()
    _, f = net(torch.tensor(x, dtype=torch.float))
    subplots = [221, 222, 223, 224]

    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for i in range(4):
        ff, xx, yy = f.cpu().numpy()[i], x[i], target[i]
        plt.subplot(subplots[i])
        plt.plot(range(0, backcast_length), xx, color='b')
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        # plt.title(f'step #{grad_step} ({i})')
    # plt.show()
    output = 'n_beats_{}.png'.format(grad_step)
    plt.savefig(output)
    plt.clf()

    print('Saved image to {}.'.format(output))


if __name__ == '__main__':
    main()
