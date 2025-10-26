import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm


def get_resnet18_feature_extractor():
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def compute_resnet_embeddings(
    frames_dir,
    out_path="output/embeddings_resnet.npy",
    batch_size=16,
    num_threads=4,
    device="cpu",
):
    torch.set_num_threads(num_threads)
    model = get_resnet18_feature_extractor().to(device)
    paths = sorted(list(Path(frames_dir).glob("frame_*.jpg")))
    imgs = [Image.open(p).convert("RGB") for p in paths]
    tensors = [_preprocess(img) for img in imgs]
    dataset = torch.stack(tensors, dim=0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ResNet embedding"):
            batch = batch.to(device)
            out = model(batch)
            out = out.view(out.size(0), -1)
            out = out / (out.norm(dim=1, keepdim=True) + 1e-10)
            embeddings.append(out.cpu().numpy())
    embeddings = np.vstack(embeddings).astype(np.float32)
    np.save(out_path, embeddings)
    print(f"[INFO] Saved ResNet embeddings to {out_path} shape={embeddings.shape}")
    return out_path


if __name__ == "__main__":
    compute_resnet_embeddings("output/frames", out_path="output/embeddings_resnet.npy")
