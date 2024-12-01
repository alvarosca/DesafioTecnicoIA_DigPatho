import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='PyTorch FashionMNIST Training')

    # General options
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--summary', dest='summary', action='store_true',
                        help='show net summary ')
    parser.add_argument('--dont_display_progress_bar', '-no_bar', action='store_true',
                        help='dont show progress bar')
    parser.add_argument('--test_model', '-test', action='store_true', help='tests model from checkpoint')

    parser.add_argument('--ckpt_file', '-ckpt', default='./checkpoint/ckpt.pth', type=str, help='model checkpoint file')

    # Model options
    parser.add_argument('--architecture', '-arch', default='ResNet', type=str, help='model architecture')

    # Optimization options
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    # Data augmentation options
    parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')

    parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')

    parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

    return parser.parse_args()
