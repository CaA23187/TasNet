import torch
from torch import nn
# from tensorboardX import SummaryWriter
import time
import os

from MyDataset_HDF5 import MyDataset
from conv_tasnet import ConvTasNet
from conv_tasnet import Encoder
from conv_tasnet import Decoder

# class CustomLoss(torch.nn.Module):
#     '''
#     parameter: gamma: regularization constant, 0 <= gamma <= 1
#     input:  y1: magnitude spctra of sorce 1
#                 shape (batch_size, time_step, input_size)
#             y2: magnitude spctra of sorce 2
#             y1_hat: estimated magnitude spctra of sorce 1
#             y2_hat: estimated magnitude spctra of sorce 2
#     '''
#
#     def __init__(self, gamma):
#         super(CustomLoss, self).__init__()
#         self.gamma = gamma
#
#     def forward(self, y1, y2, y1_hat, y2_hat):
#         loss = torch.sum(
#             (y1 - y1_hat) ** 2 + (y2 - y2_hat) ** 2 - self.gamma * ((y1 - y2_hat) ** 2 + (y2 - y1_hat) ** 2)) / y1.size(
#             0)
#         #         loss = torch.sum((y1 - y1_hat)**2 + (y2 - y2_hat)**2)
#         return loss

class CustomLoss(torch.nn.Module):
    '''
    parameter: gamma: regularization constant, 0 <= gamma <= 1
    input:  y1: magnitude spctra of sorce 1
                shape (batch_size, time_step, input_size)
            y2: magnitude spctra of sorce 2
            y1_hat: estimated magnitude spctra of sorce 1
            y2_hat: estimated magnitude spctra of sorce 2
    '''

    def __init__(self, gamma):
        super(CustomLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y1, y2, y1_hat, y2_hat):
        s_target1 = torch.sum(y1*y1_hat, 0) * y1 / torch.sum(y1**2)
        e_noise1 = y1_hat - y1
        s_target2 = torch.sum(y2*y2_hat, 0) * y2 / torch.sum(y2**2)
        e_noise2 = y2_hat - y2
        loss = 10*torch.log10(torch.sum(e_noise1**2) / torch.sum(s_target1**2)) + 10*torch.log10(torch.sum(e_noise2**2) / torch.sum(s_target2**2))
        return loss


# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 2048
LR = 0.001
if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE=', DEVICE, "| PyTorch", torch.__version__)

    cPath = os.getcwd()  # current path
    workspace_dir = os.path.join(cPath, 'workspace')
    hdf5_dir = os.path.join(workspace_dir, 'spectra.hdf5')
    train_data = MyDataset(hdf5_dir, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_data = MyDataset(hdf5_dir, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    print(train_data.len)

    # M, T = 1024, 96
    # K = 2*T//L-1
    N, L, B, H, P, X, R, C, norm_type, causal = 256, 40, 256, 512, 3, 8, 4, 2, "gLN", False
    TasNet = ConvTasNet(N, L, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear='relu').to(DEVICE)
    TasNet.load_state_dict(torch.load('./workspace/Tasnet_params.pkl'))
    optimizer = torch.optim.Adam(TasNet.parameters(), lr=LR)
    # loss_func = CustomLoss(0)
    loss_func = nn.MSELoss()
    # writer = SummaryWriter()

    t1 = time.time()
    for epoch in range(EPOCH):
        for step, (s1, s2, mix) in enumerate(train_loader):  # gives batch data
            s1 = s1.to(DEVICE)
            s2 = s2.to(DEVICE)
            mix = mix.to(DEVICE)

            out = TasNet(mix)  # rnn DRNN

            # loss = loss_func(s1, s2, out[:,0,:], out[:,1,:])  # cross entropy loss
            loss = 10*loss_func(s1, out[:,0,:])
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 100 == 0:
                print('epoch ', epoch, 'round ' + str(step) + ' ' + str(loss.item()), 'Time spend: ', time.time() - t1)
                t1 = time.time()
                # torch.save(TasNet.state_dict(), './workspace/Tasnet_params.pkl')
                # print('save model successfully')
            # writer.add_scalar('Train Loss', loss, epoch)

        # testing
        with torch.no_grad():
            test_loss = 0
            for step, (s1, s2, mix) in enumerate(test_loader):  # gives batch data
                s1 = s1.to(DEVICE)
                s2 = s2.to(DEVICE)
                mix = mix.to(DEVICE)

                out = TasNet(mix)  # rnn DRNN

                # test_loss += loss_func(s1, s2, out[:,0,:], out[:,1,:]).item()  # cross entropy loss
                test_loss += loss_func(s1, out[:,0,:]).item()
            test_loss /= step
            print(test_loss)
#             writer.add_scalar('Test Loss', test_loss, epoch)
#
        torch.save(TasNet.state_dict(), './workspace/Tasnet_params.pkl')
        print('save model successfully')


