import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

# === CONFIGURATION ===
data_path = "/user/louay.hamdi/u13592/.project/dir.project/NSCLC_finetuning_dataset_split/test"
save_dir = "tsne_fake_outputs"
os.makedirs(save_dir, exist_ok=True)

# === LOAD LABELS ONLY ===
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = datasets.ImageFolder(root=data_path, transform=transform)
labels = [sample[1] for sample in dataset]
labels = np.array(labels)
n_samples = len(labels)
print(f"📊 Loaded {n_samples} samples.")

# === SIMULATE FEATURES BASED ON LABELS ===
def simulate_features(separation=2.0, noise=0.4):
    base = np.random.randn(n_samples, 128) * noise
    for i in range(n_samples):
        base[i] += separation if labels[i] == 0 else -separation
    return base

# === RUN t-SNE AND SAVE PLOTS ===
def run_tsne(features, seed, title, filename):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, random_state=seed)
    reduced = tsne.fit_transform(features)
    x_min, x_max = reduced.min(0), reduced.max(0)
    reduced = (reduced - x_min) / (x_max - x_min)

    plt.figure(figsize=(14, 14))
    for i in range(reduced.shape[0]):
        plt.text(reduced[i, 0], reduced[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]), fontdict={'weight': 'bold', 'size': 6})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=200)
    plt.close()

# === THREE VARIANTS ===
run_tsne(simulate_features(separation=3.0, noise=0.2), seed=42, title="Excellent Separation", filename="tsne_best.png")
run_tsne(simulate_features(separation=1.5, noise=0.6), seed=23, title="Moderate Separation", filename="tsne_mid.png")
run_tsne(simulate_features(separation=0.5, noise=1.0), seed=7, title="Poor Separation", filename="tsne_bad.png")

print("✅ All t-SNE plots saved in:", save_dir)
