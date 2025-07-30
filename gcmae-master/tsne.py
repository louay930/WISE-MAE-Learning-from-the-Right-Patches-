import argparse
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
assert timm.__version__ == "0.3.2"
from timm.models.layers import trunc_normal_
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
import models_encoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('GCMAE feature representation visual', add_help=False)
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--random', default=False)
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--save_path', default='')
    parser.add_argument('--data_path_val', default='', type=str)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--gpu_id', default=0, type=int)
    return parser

def main(args):
    torch.cuda.set_device(args.gpu_id)
    print('job dir:', os.path.dirname(os.path.realpath(__file__)))
    print(args)

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6790435, 0.5052883, 0.66902906],
                             std=[0.19158737, 0.2039779, 0.15648715])
    ])

    dataset_val = datasets.ImageFolder(args.data_path_val, transform=transform_val)
    print(f"Detected classes: {dataset_val.classes}")
    class_names = dataset_val.classes

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem,
        drop_last=False)

    model = models_encoder.__dict__[args.model](global_pool=args.global_pool)

    if args.finetune and not args.random:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Loading checkpoint:", args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k}")
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=0.01)

    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
                                     model.head)
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)
    evaluate(data_loader_val, model, device, args, class_names)

def evaluate(data_loader, model, device, args, class_names):
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30, n_iter=4000)
    model.eval()
    all_outputs, all_targets = [], []

    for batch in misc.MetricLogger().log_every(data_loader, 10, "Test:"):
        images, targets = batch[0].to(device), batch[-1].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(images)
        all_outputs.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    X = np.concatenate(all_outputs, axis=0)
    y = np.concatenate(all_targets, axis=0)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    print("t-SNE shape:", X_norm.shape)

    plt.figure(figsize=(12, 12))
    cmap = plt.get_cmap("tab10")

    for class_idx, class_name in enumerate(class_names):
        idxs = (y == class_idx)
        plt.scatter(X_norm[idxs, 0], X_norm[idxs, 1],
                    color=cmap(class_idx), label=class_name, alpha=0.6, s=8)

    plt.legend(loc='best', fontsize=10)
    plt.title("t-SNE of NCT-CRC Validation Features", fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(args.save_path)
    plt.show()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
