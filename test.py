import torch

from conv_tasnet import ConvTasNet

N, L, B, H, P, X, R, C, norm_type, causal = 256, 40, 256, 512, 3, 7, 2, 2, "gLN", False
TasNet = ConvTasNet(N, L, B, H, P, X, R, C, norm_type="gLN", causal=True, mask_nonlinear='relu')
a = torch.randn(3, 96)

b = TasNet(a)
print(b)

torch.save(TasNet.state_dict(), './workspace/Tasnet_params.pkl')

net2 = ConvTasNet(N, L, B, H, P, X, R, C, norm_type="gLN", causal=True, mask_nonlinear='relu')
net2.load_state_dict(torch.load('./workspace/Tasnet_params.pkl'))
c = net2(a)
print(c)
