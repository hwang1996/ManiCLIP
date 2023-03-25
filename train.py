import argparse
import os
import pickle
import numpy as np
import random
import warnings
import time
from typing import Optional

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn import LayerNorm
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import pickle
import math

import clip
import numpy as np
import shutil

from external.stylegan2.model import Generator
from external.stylegan2.calc_inception import load_patched_inception_v3

from utils.id_loss import IDLoss
from utils.utils import adjust_learning_rate,AverageMeter,ProgressMeter,calc_fid,int_item,parse_mask
from utils.average_lab_color_loss import AvgLabLoss
from utils.data_processing import produce_labels
from utils.model_irse import IRSE


parser = argparse.ArgumentParser(description='ManiCLIP Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=31, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='training mini-batch size')
parser.add_argument('--test_batch', default=50, type=int, metavar='N',
                    help='test batchsize (default: 50)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('-p', '--print-freq', default=30, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=10, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',  
                    help='evaluate model on validation set')
parser.add_argument('--task_name', default='model', type=str,
                    help='task name')
parser.add_argument('--loss_clip_weight', type=float, default=1.0,
                    help='The clip loss for optimization. (default: 2.0)')
parser.add_argument('--loss_w_norm_weight', type=float, default=0.01,
                    help='The L2 loss on the latent codes w for optimization. (default: 0.01)')
parser.add_argument('--loss_minmaxentropy_weight', type=float, default=0,
                    help='The entropy loss on the latent codes w for optimization. (default: 0.)')
parser.add_argument('--loss_id_weight', type=float, default=0,
                    help='The ID loss for optimization. (default: 0.3)')
parser.add_argument('--loss_face_bg_weight', type=float, default=0.,
                    help='The face background loss for optimization. (default: 1.0)')
parser.add_argument('--loss_face_norm_weight', type=float, default=0.,
                    help='The loss between the input and output face area colous. (default: 0.)')
parser.add_argument("--truncation", type=float, default=0.7, help="truncation ratio")
parser.add_argument(
    "--truncation_mean",
    type=int,
    default=4096,
    help="number of vectors to calculate mean for the truncation",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="pretrained/ffhq_256.pt",
    help="path to the model checkpoint",
)
parser.add_argument(
    "--size", type=int, default=256, help="output image size of the generator"
)
parser.add_argument('--decouple', action='store_true',  
                    help='Use decoupling training scheme')
parser.add_argument(
    "--part_sample_num", default=3, type=int, help="the number of attributes sampled for each text segment"
)

class CLIPLoss(nn.Module):
    def __init__(self, clip_model):
        super(CLIPLoss, self).__init__()
        self.model = clip_model

    def forward(self, image, text):
        image = torchvision.transforms.functional.resize(image, 224)
        distance = 1 - self.model(image, text)[0] / 100
        return distance

def init_parsing_model(args):
    from external.parsing import BiSeNet
    args.parse_model = BiSeNet(n_classes=19)
    args.parse_model.load_state_dict(torch.load('external/parsing/models/bisenet.pth'))

    args.parse_model = args.parse_model.cuda()
    args.parse_model.eval()

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu

    args.save_folder = os.path.join('models', args.task_name)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    args.clip_model, _ = clip.load("ViT-B/32", device="cuda")
    clip_loss = CLIPLoss(args.clip_model)
    args.id_loss = IDLoss().cuda().eval()

    face_model = IRSE()
    face_model = nn.DataParallel(face_model).cuda()
    checkpoint = torch.load('pretrained/attribute_model.pth.tar')
    face_model.load_state_dict(checkpoint['state_dict'])
    face_model.eval()

    args.face_model = face_model

    # create model
    model = TransModel(nhead=8, num_decoder_layers=6)
    model.clip_model = model.clip_model.float()
    print(model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.5, 0.999))
    
    # optionally resume from a checkpoint
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        load_path = os.path.join(args.resume)
        if args.gpu is None:
            checkpoint = torch.load(load_path)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(load_path, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.decouple:
        dataset_train = PartTextDataset(split='train', sample_num=args.part_sample_num)
    else:
        dataset_train = TextDataset(split='train')
    train_loader = torch.utils.data.DataLoader(dataset_train, 
                                                batch_size=args.batch_size, 
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True,
                                                drop_last=True)
    dataset_eval = TextDataset(split='eval')
    eval_loader = torch.utils.data.DataLoader(dataset_eval, 
                                                batch_size=args.test_batch, 
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True,
                                                drop_last=False)

    generator = Generator(args.size, 512, 8).cuda()
    generator.load_state_dict(torch.load(args.ckpt)['g_ema'], strict=False)
    generator.eval()

    with open('pretrained/inception_ffhq.pkl', 'rb') as f:
        embeds = pickle.load(f)
        args.real_mean = embeds['mean']
        args.real_cov = embeds['cov']

    args.inception = nn.DataParallel(load_patched_inception_v3()).cuda()
    args.inception.eval()

    args.ce_noreduced = nn.CrossEntropyLoss(reduce=False).cuda()
    args.ce_criterion = nn.CrossEntropyLoss().cuda()

    args.average_color_loss = AvgLabLoss().cuda().eval()

    fid_best = float("inf")
    init_parsing_model(args)

    if args.truncation < 1:
        with torch.no_grad():
            args.mean_latent = generator.mean_latent(args.truncation_mean)
    else:
        args.mean_latent = None
    
    if args.evaluate:
        with torch.no_grad():
            fid = validate(eval_loader, model, None, generator, clip_loss, 0, args)
        return        

    log_path = os.path.join('logs', args.task_name)
    writter = SummaryWriter(log_path)

    iteration_num = args.start_epoch*len(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        train(train_loader, model, writter, generator, clip_loss, optimizer, epoch, args, iteration_num=iteration_num)
        iteration_num += len(train_loader)
        
        with torch.no_grad():
            fid = validate(eval_loader, model, writter, generator, clip_loss, epoch, args)
        if fid < fid_best:
            fid_best = fid
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=True, save_folder=args.save_folder)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, save_folder=args.save_folder)
       
def save_checkpoint(state, is_best, save_folder, filename='latest.pth.tar'):
    filename = os.path.join(save_folder, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_folder, 'model_best.pth.tar'))

def train(train_loader, model, writter, generator, clip_loss, optimizer, epoch, args, iteration_num=0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    clip_losses = AverageMeter('clip_loss', ':.4e')
    bg_losses = AverageMeter('bg_loss', ':.4e')
    w_norm_losses = AverageMeter('w_norm_loss', ':.4e')
    id_losses = AverageMeter('id_loss', ':.4e')
    face_norm_losses = AverageMeter('face_norm_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    all_losses = AverageMeter('all_losses', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, all_losses, clip_losses, w_norm_losses, id_losses, face_norm_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (clip_text, sampled_text, labels, exist_mask, length) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            clip_text       = clip_text.cuda(args.gpu, non_blocking=True)
            labels          = labels.cuda(args.gpu, non_blocking=True)
            exist_mask      = exist_mask.cuda(args.gpu, non_blocking=True)
            
        with torch.no_grad():
            code = torch.randn(args.batch_size, 512).cuda()
            styles = generator.style(code)
            input_im, _ = generator([styles], input_is_latent=True, randomize_noise=False, 
                        truncation=args.truncation, truncation_latent=args.mean_latent)

        offset     = model(styles, clip_text)
        new_styles = styles.unsqueeze(1).repeat(1, 14, 1) + offset

        gen_im, _ = generator([new_styles], input_is_latent=True, randomize_noise=False, 
                        truncation=args.truncation, truncation_latent=args.mean_latent)

        input_im = input_im.clamp(min=-1,max=1)
        gen_im = gen_im.clamp(min=-1,max=1)
        
        loss = 0.0
        if args.loss_face_bg_weight:
            input_im_mask_hair, input_im_mask_face = parse_mask(args, input_im)
            input_im_bg_mask = ((input_im_mask_hair + input_im_mask_face)==0).float()
            gen_im_mask_hair, gen_im_mask_face = parse_mask(args, gen_im)
            gen_im_bg_mask = ((gen_im_mask_hair + gen_im_mask_face)==0).float()
            bg_mask = ((input_im_bg_mask+gen_im_bg_mask)==2).float()

            loss_bg = torch.mean((input_im*bg_mask - gen_im*bg_mask) ** 2)
            loss = loss + loss_bg * args.loss_face_bg_weight
            bg_losses.update(loss_bg.item(), styles.size(0))
            writter.add_scalar('Train/Face BG loss', bg_losses.avg, iteration_num+i)

        if args.loss_id_weight:
            loss_id = args.id_loss(gen_im, input_im)

            loss = loss + loss_id * args.loss_id_weight
            id_losses.update(loss_id.item(), styles.size(0))
            writter.add_scalar('Train/ID loss', id_losses.avg, iteration_num+i)

        if args.loss_face_norm_weight:
            _, input_im_mask_face = parse_mask(args, input_im)
            _, gen_im_mask_face = parse_mask(args, gen_im)
            loss_face_norm = args.average_color_loss(gen_im, input_im, gen_im_mask_face, input_im_mask_face)
            loss = loss + loss_face_norm * args.loss_face_norm_weight
            face_norm_losses.update(loss_face_norm.item(), styles.size(0))
            writter.add_scalar('Train/Face norm loss', face_norm_losses.avg, iteration_num+i)

        if args.loss_w_norm_weight:
            loss_latent_norm = torch.mean(offset ** 2)
            loss = loss + loss_latent_norm * args.loss_w_norm_weight
            w_norm_losses.update(loss_latent_norm.item(), styles.size(0))
            writter.add_scalar('Train/W norm loss', w_norm_losses.avg, iteration_num+i)

        if args.loss_minmaxentropy_weight:
            offset = offset.reshape(offset.size(0), -1)
            offset = offset.abs()
            offset_max = torch.max(offset, 1)[0].unsqueeze(1)
            offset_min = torch.min(offset, 1)[0].unsqueeze(1)
            offset_p = (offset - offset_min) / (offset_max - offset_min) + 1e-7

            pseudo_entropy_loss = (- (offset_p * torch.log(offset_p)).sum(1).mean()) * 0.0001

            loss = loss + args.loss_minmaxentropy_weight * pseudo_entropy_loss
            entropy_losses.update(pseudo_entropy_loss.item(), styles.size(0))
            writter.add_scalar('Train/Entropy loss', entropy_losses.avg, iteration_num+i)

        # CLIP loss.
        if args.loss_clip_weight:
            loss_clip = clip_loss(gen_im, clip_text)
            loss_clip = torch.diag(loss_clip).mean()
            loss = loss + loss_clip * args.loss_clip_weight 
            clip_losses.update(loss_clip.item(), styles.size(0))
            writter.add_scalar('Train/CLIP loss', clip_losses.avg, iteration_num+i)
        
        all_losses.update(loss.item(), styles.size(0))
        writter.add_scalar('Train/all loss', all_losses.avg, iteration_num+i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            ##### for visualization
            vis_ = make_grid(gen_im[:9].clamp(min=-1,max=1)*0.5+0.5, nrow=3, normalize=False)
            save_path = os.path.join(args.save_folder, 'out_face')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torchvision.utils.save_image(vis_, os.path.join(save_path, str(epoch)+'.png'))

            vis_ = make_grid(input_im[:9].clamp(min=-1,max=1)*0.5+0.5, nrow=3, normalize=False)
            save_path = os.path.join(args.save_folder, 'in_face')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torchvision.utils.save_image(vis_, os.path.join(save_path, str(epoch)+'.png'))

            save_path = os.path.join(args.save_folder, 'text')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            n = open(os.path.join(save_path, str(epoch)+'.txt'),'w')
            for _ in range(9):
                n.write(str(_)+': '+sampled_text[_]+'\n')
            n.close()
    
    return clip_losses.avg

def validate(eval_loader, model, writter, generator, clip_loss, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    clip_losses = AverageMeter('clip_loss', ':.4e')
    w_norm_losses = AverageMeter('w_norm_loss', ':.4e')
    bg_losses = AverageMeter('bg_loss', ':.4e')
    id_losses = AverageMeter('id_loss', ':.4e')
    face_norm_losses = AverageMeter('face_norm_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    all_losses = AverageMeter('all_losses', ':.4e')
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, data_time, all_losses, clip_losses, w_norm_losses, id_losses, face_norm_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to eval mode
    model.eval()

    acc_avg = AverageMeter()

    features = []
    end = time.time()
    for i, (clip_text, sampled_text, labels, exist_mask, length, test_latents) in enumerate(eval_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            clip_text           = clip_text.cuda(args.gpu, non_blocking=True)
            labels              = labels.cuda(args.gpu, non_blocking=True)
            exist_mask          = exist_mask.cuda(args.gpu, non_blocking=True)
            test_latents        = test_latents.cuda(args.gpu, non_blocking=True)
    
        code = test_latents
        styles = generator.style(code)
        input_im, _ = generator([styles], input_is_latent=True, randomize_noise=False, 
                    truncation=args.truncation, truncation_latent=args.mean_latent)

        offset     = model(styles, clip_text)
        new_styles = styles.unsqueeze(1).repeat(1, 14, 1) + offset
        
        gen_im, _ = generator([new_styles], input_is_latent=True, randomize_noise=False, 
                        truncation=args.truncation, truncation_latent=args.mean_latent)

        input_im = input_im.clamp(min=-1,max=1)
        gen_im = gen_im.clamp(min=-1,max=1)

        in_attr = args.face_model(torchvision.transforms.functional.resize(input_im, 256))
        gen_attr = args.face_model(torchvision.transforms.functional.resize(gen_im, 256))
        in_preds = torch.stack(in_attr).transpose(0, 1).argmax(-1)
        gen_preds = torch.stack(gen_attr).transpose(0, 1).argmax(-1)
        out_label = torch.where(exist_mask==1, labels.long(), in_preds)
        acc = (((gen_preds == out_label).sum(1) / gen_preds.size(1)).mean().item()) * 100
        acc_avg.update(acc, styles.size(0))

        feat = args.inception(gen_im)[0].view(gen_im.shape[0], -1)
        features.append(feat.to('cpu'))

        loss = 0.0
        if args.loss_face_bg_weight:
            input_im_mask_hair, input_im_mask_face = parse_mask(args, input_im)
            input_im_bg_mask = ((input_im_mask_hair + input_im_mask_face)==0).float()
            gen_im_mask_hair, gen_im_mask_face = parse_mask(args, gen_im)
            gen_im_bg_mask = ((gen_im_mask_hair + gen_im_mask_face)==0).float()
            bg_mask = ((input_im_bg_mask+gen_im_bg_mask)==2).float()

            loss_bg = torch.mean((input_im*bg_mask - gen_im*bg_mask) ** 2)
            loss = loss + loss_bg * args.loss_face_bg_weight
            bg_losses.update(loss_bg.item(), styles.size(0))
            if i == len(eval_loader)-1 and writter != None:
                writter.add_scalar('Val/Face BG loss', bg_losses.avg*100, epoch)

        if args.loss_id_weight:
            loss_id = args.id_loss(gen_im, input_im)

            loss = loss + loss_id * args.loss_id_weight
            id_losses.update(loss_id.item(), styles.size(0))
            if i == len(eval_loader)-1 and writter != None:
                writter.add_scalar('Val/ID loss', id_losses.avg*100, epoch)

        if args.loss_face_norm_weight:
            _, input_im_mask_face = parse_mask(args, input_im)
            _, gen_im_mask_face = parse_mask(args, gen_im)
            loss_face_norm = args.average_color_loss(gen_im, input_im, gen_im_mask_face, input_im_mask_face)
            loss = loss + loss_face_norm * args.loss_face_norm_weight
            face_norm_losses.update(loss_face_norm.item(), styles.size(0))
            if i == len(eval_loader)-1 and writter != None:
                writter.add_scalar('Val/Face norm loss', face_norm_losses.avg*100, epoch)
            
        if args.loss_w_norm_weight:
            loss_latent_norm = torch.mean(offset ** 2)
            loss = loss + loss_latent_norm * args.loss_w_norm_weight
            w_norm_losses.update(loss_latent_norm.item(), styles.size(0))
            if i == len(eval_loader)-1 and writter != None:
                writter.add_scalar('Val/W norm loss', w_norm_losses.avg*100, epoch)

        if args.loss_minmaxentropy_weight:
            offset = offset.reshape(offset.size(0), -1)
            offset = offset.abs()
            offset_max = torch.max(offset, 1)[0].unsqueeze(1)
            offset_min = torch.min(offset, 1)[0].unsqueeze(1)
            offset_p = (offset - offset_min) / (offset_max - offset_min) + 1e-7

            pseudo_entropy_loss = (- (offset_p * torch.log(offset_p)).sum(1).mean()) * 0.0001

            loss = loss + args.loss_minmaxentropy_weight * pseudo_entropy_loss
            entropy_losses.update(pseudo_entropy_loss.item(), styles.size(0))
            if i == len(eval_loader)-1 and writter != None:
                writter.add_scalar('Val/Entropy loss', entropy_losses.avg, epoch)

        # CLIP loss.
        if args.loss_clip_weight:
            loss_clip = clip_loss(gen_im, clip_text)
            loss_clip = torch.diag(loss_clip).mean()
            loss = loss + loss_clip * args.loss_clip_weight
            clip_losses.update(loss_clip.item(), styles.size(0))
            if i == len(eval_loader)-1 and writter != None:
                writter.add_scalar('Val/CLIP loss', clip_losses.avg*100, epoch)

        all_losses.update(loss.item(), styles.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            ##### for visualization
            vis_ = make_grid(gen_im[:9].clamp(min=-1,max=1)*0.5+0.5, nrow=3, normalize=False)
            save_path = os.path.join(args.save_folder, 'eval_out_face')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torchvision.utils.save_image(vis_, os.path.join(save_path, str(epoch)+'.png'))

            vis_ = make_grid(input_im[:9].clamp(min=-1,max=1)*0.5+0.5, nrow=3, normalize=False)
            save_path = os.path.join(args.save_folder, 'eval_in_face')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torchvision.utils.save_image(vis_, os.path.join(save_path, str(epoch)+'.png'))

            save_path = os.path.join(args.save_folder, 'eval_text')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            n = open(os.path.join(save_path, str(epoch)+'.txt'),'w')
            for _ in range(9):
                n.write(str(_)+': '+sampled_text[_]+'\n')
            n.close()

    features = torch.cat(features, 0).numpy()
    print(f'extracted {features.shape[0]} features')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    fid = calc_fid(sample_mean, sample_cov, args.real_mean, args.real_cov)
    if writter != None:
        writter.add_scalar('Val/all loss', all_losses.avg, epoch)
        writter.add_scalar('Val/fid', fid, epoch)
        writter.add_scalar('Val/face acc', acc_avg.avg, epoch)
        writter.add_scalar('Val/face id sim', 100*(1-id_losses.avg), epoch)
    print('fid: ', fid)
    print('face id sim: ', 100*(1-id_losses.avg))
    print('face attribute accuracy: ', acc_avg.avg)

    return fid

class TextDataset(data.Dataset):
    def __init__(self, split='train'):
        self.text_dir = 'data/celeba-caption/'
        self.text_files = os.listdir(self.text_dir)
        self.text_files.sort(key=int_item)
        f = open('data/list_attr_celeba.txt')
        data = f.readlines()
        attrs = data[1].split(' ')
        attrs[-1] = attrs[-1][:-1]
        self.attrs = np.array([' '.join(a.split('_')).lower() for a in attrs], dtype=object)
        self.anno = data[2:]
        train_num = 25000
        if split == 'train':
            self.text_files = self.text_files[:train_num]
            self.anno = self.anno[:train_num]
        else:
            self.text_files = self.text_files[train_num:]
            self.anno = self.anno[train_num:]

            self.test_latents = torch.load('data/test_latents_seed100.pt')

        self.split = split

        self.non_represents = ['no', 'hair', 'wearing', 'eyebrows', 'eyes', 'big', 'nose', 'o']
        self.gender_list = ['he', 'she', 'man', 'woman']

    def __len__(self):
        return len(self.text_files)
        
    def __getitem__(self, index):
        text_filename = self.text_files[index]
        text_path = os.path.join(self.text_dir, text_filename)
        text_set = open(text_path).readlines()
        
        sampled_text = text_set[0][:-1]
        
        anno = self.anno[index][:-1].split(' ')[1:]
        
        clip_text, labels, exist_mask = produce_labels(sampled_text, anno, self.attrs, self.gender_list, self.non_represents)
        
        length = torch.where(clip_text==0)[1][0].item()
        if self.split == 'train':
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length
        else:
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length, self.test_latents[index]

class PartTextDataset(data.Dataset):
    def __init__(self, split='train', sample_num=3):
        self.test_latents = torch.load('data/test_latents_seed100.pt')

        self.split = split
        self.sample_num = sample_num

        f = open('data/list_attr_celeba.txt')
        self.data = f.readlines()
        attrs = self.data[1].split(' ')
        attrs[-1] = attrs[-1][:-1]
        self.attrs = np.array([' '.join(a.split('_')).lower() for a in attrs], dtype=object)

        self.img_attr = self.data[2:25002]

        self.hair = ['bald', 'bangs', 'black hair', 'blond hair', 'brown hair', 'gray hair', 'receding hairline', 'straight hair', 'wavy hair']
        self.eye = ['arched eyebrows', 'bags under eyes', 'bushy eyebrows', 'eyeglasses', 'narrow eyes']
        self.fashion = ['attractive', 'heavy makeup', 'high cheekbones', 'rosy cheeks', 'wearing earrings', 'wearing hat', 'wearing lipstick', 'wearing necklace', 'wearing necktie']
        self.others = ['5 o clock shadow', 'big nose', 'blurry', 'chubby', 'double chin', 'no beard', 'oval face', 'pale skin', 'pointy nose', 'young']
        self.mouth = ['big lips', 'mouth slightly open', 'smiling', 'goatee', 'mustache', 'sideburns']

        self.groups = [self.hair, self.eye, self.fashion, self.others, self.mouth]

    def __len__(self):
        return len(self.img_attr)
        
    def __getitem__(self, index):

        sampled_class = torch.randint(0, 5, (1,)).item()
        instance_attr = self.groups[sampled_class]
        sampled_cate = torch.randperm(len(instance_attr))[:self.sample_num]
        attr = np.array(instance_attr)[sampled_cate]
        if self.sample_num == 1:
            attr = np.array([attr])
        selected_cate_40 = []
        for x in attr:
            selected_cate_40.append(np.where(self.attrs==x)[0][0])

        gender = torch.randint(0, 3, (1,)).item()
        concat_text = ', '.join(attr)
        if gender == 0:
            sampled_text = 'she has ' + concat_text
        elif gender == 1:
            sampled_text = 'he has ' + concat_text
        else:
            sampled_text = 'the person has ' + concat_text

        clip_text = clip.tokenize(sampled_text)
        exist_mask = torch.zeros(40)
        exist_mask[selected_cate_40] = 1
        labels = exist_mask.clone()

        if gender != 2:
            exist_mask[20] = 1
            if gender == 1:
                labels[20] = 1
            else:
                labels[20] = 0

        length = torch.where(clip_text==0)[1][0].item()
        if self.split == 'train':
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length
        else:
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length, self.test_latents[index]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransModel(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 4, dim_feedforward: int = 2048, 
                activation: str = "relu", dropout: float = 0.2,
                num_decoder_layers: int = 4,):
        super(TransModel, self).__init__()
        
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.transform = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.face_pool = nn.AdaptiveAvgPool2d((224, 224))

        self.pos_encoder = PositionalEncoding(d_model, max_len=20)
        self.x_map = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
        )
        self.text_map = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
        )
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    def forward(self, x, text_inputs):
        
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_inputs).detach()
        text_embedding = self.text_map(text_embedding)
        text_embedding = text_embedding + self.text_map(text_embedding)
        text_embedding = self.norm1(text_embedding)

        x = x.unsqueeze(1).repeat(1, 14, 1).transpose(0, 1)                 # [seq_len, batch_size, embedding_dim]
        x = self.pos_encoder(x)                                             # [seq_len, batch_size, embedding_dim]
        x = self.x_map(x)                                                 
        x = x + self.x_map(x)                                                  
        x = self.norm2(x)

        out = self.decoder(tgt=x, memory=text_embedding.unsqueeze(1).transpose(0, 1))
        
        out = out.transpose(0, 1)

        return out


if __name__ == "__main__":
    main()
    
