import torch.optim as optim
from models.loss_functions import LSALoss
from models import LSAMNIST
from torch.utils.data import DataLoader
from datasets import MNIST
from torch import save

for normal_class in range(10):
  # Build dataset and model
  model = LSAMNIST(input_shape=(1, 28, 28), code_length=64, cpd_channels=100).cuda().eval()
  loss = LSALoss(cpd_channels=100, lam=1)
  optimizer = optim.Adam(model.parameters(), lr=10**-4)

  trainset = MNIST(path='data/MNIST')
  trainset.val(normal_class)
  trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=1)

  print(f'Training {trainset}')
  for epoch in range(200):  # loop over the dataset multiple times
    for i, (x, y) in enumerate(trainloader):
      # get the inputs
      x = x.to('cuda')

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      x_r, z, z_dist = model(x)
      loss(x, x_r, z, z_dist).backward()
      optimizer.step()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, loss.total_loss))

  save(model.state_dict(), f'checkpoints/mnist2/{normal_class}.pkl')
  print('Finished Training')
