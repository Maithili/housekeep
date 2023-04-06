import torch
import clip
from PIL import Image

def test_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(["a cat", "a black cat", "a black cat on the couch", "a black pan"]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        print(text_features.shape)
        text_feature_norms = torch.linalg.norm(text_features, dim=1, keepdim=True)
        print(text_feature_norms)

        probs = torch.matmul(text_features, text_features.T)/torch.matmul(text_feature_norms, text_feature_norms.T)

    print("Probability Matrix:\n", probs)

if __name__ == '__main__':
    test_clip()