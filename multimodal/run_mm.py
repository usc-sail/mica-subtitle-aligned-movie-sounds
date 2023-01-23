### Author: Rajat Hebbar
### Borrowed from AST Code, credit Yuon Gong: https://github.com/YuanGongND/ast.git 

import argparse
import os
import ast
import pickle
import sys
import time
import torch
sys.path.append('../')
import dataloader_mm
import models
import numpy as np
from traintest_mm import train, validate

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n-class", type=int, default=527, help="number of classes")
parser.add_argument("--posembed", type=str, default=None, help="position embedding for clip")
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')
parser.add_argument('--movieset_pretrain', help='if use Movie Audio Events pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

args = parser.parse_args()


print('now train a audio spectrogram transformer model')
# dataset spectrogram mean and std, used to normalize the input
spec_mean = -5.2091117
spec_std = 3.8955019

audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': 'movie_sounds', 'mode':'train', 'mean':spec_mean, 'std':spec_std}
val_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'movie_sounds', 'mode':'evaluation', 'mean':spec_mean 'std':spec_std}

train_loader = torch.utils.data.DataLoader(
    dataloader_clipmm.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, n_class=args.n_class),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloader_clipmm.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf, n_class=args.n_class),
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

audio_model = models.ASTModelMM(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                              input_tdim=1024, clip_pe= args.posembed, imagenet_pretrain=args.imagenet_pretrain,
                              audioset_pretrain=args.audioset_pretrain, movieset_pretrain=args.movieset_pretrain, 
                              model_size='base384')

print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd)

# best model on the validation set
stats, _ = validate(audio_model, val_loader, args, 'valid_set')
# note it is NOT mean of class-wise accuracy
val_acc = stats[0]['acc']
val_mAUC = np.mean([stat['auc'] for stat in stats])
val_mAP = np.mean([stat['AP'] for stat in stats if not np.isnan(stat['AP'])])
print('---------------evaluate on the validation set---------------')
print("Accuracy: {:.6f}".format(val_acc))
print("AUC: {:.6f}".format(val_mAUC))
print("mAP: {:.6f}".format(val_mAP))

# test the model on the evaluation set
eval_loader = torch.utils.data.DataLoader(
    dataloader_clipmm.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf, n_class=args.n_class),
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
eval_acc = stats[0]['acc']
eval_mAUC = np.mean([stat['auc'] for stat in stats])
eval_mAP = np.mean([stat['AP'] for stat in stats if not np.isnan(stat['AP'])])
print('---------------evaluate on the test set---------------')
print("Accuracy: {:.6f}".format(eval_acc))
print("AUC: {:.6f}".format(eval_mAUC))
print("AP: {:.6f}".format(eval_mAP))
with open("%s/test_stats_final.pkl" %args.exp_dir, "wb") as f:
    pickle.dump(stats, f)
np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, val_mAP, eval_acc, eval_mAUC, eval_mAP])

