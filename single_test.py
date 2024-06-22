import os
from glob import glob
from argparse import ArgumentParser
from torchvision import transforms
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from src.datamodules.datamodule import DrTMODataModule
from src.models.ednet import EDNet

torch.autograd.set_grad_enabled(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.set_detect_anomaly(mode=False)


def getDisplayableImg(img):
    # [0, 255]
    img = (img * 255).clamp(min=0, max=255).round()
    # uint8
    img = img.type(torch.uint8)
    # [1, 3, H, W] -> [H, W, 3]
    img = img.squeeze(0).permute(1, 2, 0)
    # To numpy
    img = img.cpu().numpy()
    return img


def inference(model, img, mask, exps, base_name, out_dir):
    for exp in exps:
        isUpExposed=True
        if exp < 1: isUpExposed=False
        pred, _ = model(img, mask, exp, isUpExposed=isUpExposed)  
        folder_dir = os.path.splitext(base_name)[0]
        filename = f"{folder_dir}_{exp}.jpg"
        pred = Image.fromarray(getDisplayableImg(pred))
        os.makedirs(os.path.join(out_dir, folder_dir), exist_ok=True)
        pred.save(os.path.join(out_dir, folder_dir, filename))
        

def single_hdr(args):
    device = torch.device("cpu") if args.cpu else torch.device("cuda")

    model = EDNet.load_from_checkpoint(args.ckpt).to(device=device)
    model.eval()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # Get all image paths in the specified directory
    img_paths = glob(os.path.join(args.img_dir, '*.jpg')) + glob(os.path.join(args.img_dir, '*.png'))

    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path)
        img_folder = os.path.basename(os.path.dirname(img_path))
        out_folder = os.path.join(args.out_dir, img_folder)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # Load the image
        img = Image.open(img_path).convert('RGB')
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)

        # Example exposure ratios, you can define your own set of exposure ratios
        exposure_ratios = [0.25, 0.3536, 0.5000, 0.7071, 1.4142, 1.0, 2.0, 2.8284, 4.0]

        # Process the image with different exposure ratios
        inference(model, img_tensor, None, exposure_ratios, img_name, args.out_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="Output directory to save processed images")
    parser.add_argument("--ckpt", required=True, help="Path to the checkpoint of the trained model")
    parser.add_argument("--img_dir", required=True, help="Directory containing input images to process")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU for inference")
    args = parser.parse_args()
    single_hdr(args)

