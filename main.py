from train import Trainer
# from tester import Tester
# from data_loader import Data_Loader
from torch.backends import cudnn
from utils.utils import make_folder
from utils.dataloader import CelebADataset
import os
import argparse


def str2bool(v):
    return v.lower() in ('true')


def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='sagan',
                        choices=['sagan', 'qgan'])
    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--version', type=str, default='sagan_1')

    # Training setting
    parser.add_argument('--max_iter', type=int, default=10000,
                        help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--supervise', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str,
                        default='cifar', choices=['lsun', 'celeb'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--image_path', type=str,
                        default='/home/minteiko/developer/project/data/celebA')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)
    parser.add_argument('--hist_step', type=float, default=1000)

    return parser.parse_args()


def main(config):
    # For fast training
    cudnn.benchmark = True

    # Data loader
    data_loader = CelebADataset(config.supervise, config.train, config.dataset, config.image_path, config.imsize,
                                batchsize=config.batchsize, shuffle=config.train)

    # Create directories if not exist
    check_path = os.path.join(config.sample_path, config.version)
    while os.path.exists(check_path):
        name, version = config.version.split("_")
        config.version = f"{name}_{int(version)+1}"
        check_path = os.path.join(config.sample_path, config.version)
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)

    if config.train:
        # if config.model == 'sagan':
        trainer = Trainer(data_loader.loader(), config)
        # elif config.model == 'qgan':
        # trainer = qgan_trainer(data_loader.loader(), config)
        trainer.train()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
