from external.stylegan2.model import Generator

import torch
from torchvision import utils
import os
import clip
import argparse

from train import TransModel

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--text", type=str, required=True, help='The input text for image editing')
  parser.add_argument("--gen_num", type=int, required=True, help='Number of generated samples')

  return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    ckpt = 'pretrained/ffhq_256.pt'
    generator = Generator(256, 512, 8).cuda()
    generator.load_state_dict(torch.load(ckpt)['g_ema'], strict=False)
    generator.eval()

    model = TransModel(nhead=8, num_decoder_layers=6).cuda()
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict['state_dict'])
    model.clip_model = model.clip_model.float()
    model.eval()

    input_text = [args.text] * args.gen_num
    clip_text = clip.tokenize(input_text).cuda()

    truncation = 0.7
    truncation_mean = 4096
    save_path = 'generation'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        mean_latent = generator.mean_latent(truncation_mean)

    code = torch.load('data/test_latents_seed100.pt')
    selected_idx = torch.randperm(len(code))[:args.gen_num]
    code = code[selected_idx].cuda()
    with torch.no_grad():
        styles = generator.style(code)
        input_im, _ = generator([styles], input_is_latent=True, randomize_noise=False, 
                        truncation=truncation, truncation_latent=mean_latent)

        offset = model(styles, clip_text)

        new_styles = styles.unsqueeze(1).repeat(1, 14, 1) + offset
        gen_im, _ = generator([new_styles], input_is_latent=True, randomize_noise=False, 
                        truncation=truncation, truncation_latent=mean_latent)

        utils.save_image(input_im, save_path+"/input.png", nrow=args.gen_num, padding=10, normalize=True, range=(-1, 1), pad_value=1)
        utils.save_image(gen_im, save_path+"/output.png", nrow=args.gen_num, padding=10, normalize=True, range=(-1, 1), pad_value=1)

