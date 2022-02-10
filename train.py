import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from utils import augment, adjust_learning_rate, optim
from model import VICRegNet
from loss import sim_loss, cov_loss, std_loss
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='VICReg Training')
parser.add_argument('--path', help='path to dataset')
parser.add_argument('--batch_size', default=2048, type=int, help='batch size')
parser.add_argument('--device', default='cuda', type=str, help='device for training')
parser.add_argument('--l', default=25,  help='coefficients of the invariance')
parser.add_argument('--mu', default=25,  help='coefficients of the variance')
parser.add_argument('--nu', default=1,  help='coefficients of the covariance')
parser.add_argument('--weight_decay', default=1e-6,  help='weight decay')
parser.add_argument('--lr', default=0.2,  help='weight decay')
parser.add_argument('--epoch', default=1000,  help='number of epochs')
parser.add_argument('--log_dir', default=r'logs', help = 'directory to save logs')
parser.add_argument('--save_chpt', default = 'checkpoints', help = 'path to save checkpoints')
parser.add_argument('--save_freq', default=1000, help='step frequency to save checkpoints')

def main():
    print('Training Starts')
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=args.log_dir)
    t_set = datasets.ImageFolder(root=args.path, transform=augment)
    loader = DataLoader(t_set, batch_size=args.batch_size)

    model = VICRegNet().to(args.device)
    optimizer = optim(model, args.weight_decay)

    os.makedirs(os.path.join('.', args.save_chpt), exist_ok=True)

    for epoch in range(0, args.epoch):
        loop = tqdm(enumerate(loader, start=epoch * len(loader)), total=len(loader), leave=False)
        for step, ((img_a, img_b), _) in loop:
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            repr_a = model(img_a.to(args.device))
            repr_b = model(img_b.to(args.device))

            _sim_loss = sim_loss(repr_a, repr_b)
            _std_loss = std_loss(repr_a, repr_b)
            _cov_loss = cov_loss(repr_a, repr_b)

            loss = args.l * _sim_loss + args.mu * _std_loss + args.nu * _cov_loss
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch)

            if step % int(args.save_freq) == 0 and step:
                with open(os.path.join(args.log_dir, 'logs.txt'), 'a') as log_file:
                    log_file.write(f'Epoch: {epoch}, Step: {step}, Train loss: {loss.cpu().detach().numpy()} \n')

                state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())

                torch.save(state, os.path.join('.', args.save_chpt, f'checkpoint_{step}_steps.pth'))
            loop.set_description(f'Epoch [{epoch}/{args.epoch}]')
            loop.set_postfix(loss = loss.cpu().detach().numpy())

        print(f'Loss for epoch {epoch} is {loss.cpu().detach().numpy()}')
    print('End of the Training. Saving final checkpoints.')
    state = dict(epoch=args.epoch, model=model.state_dict(),
                 optimizer=optimizer.state_dict())
    torch.save(state, os.path.join('.', args.save_chpt,  'final_checkpoint.pth'))
    writer.flush()
    writer.close()

if '__main__' == __name__:
    main()

