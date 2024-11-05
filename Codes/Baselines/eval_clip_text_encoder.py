from pathlib import Path

from eval_clip import *

def clip_inference_text(loaded_model, text_inputs, test_loader, device, binary=False):
    with torch.no_grad():
        img_name = []
        gt_labels = []
        scores = []

        for sampled_batch in tqdm(test_loader):
            img = sampled_batch['image']
            name = sampled_batch['name']
            label = sampled_batch['label']
            text_description = sampled_batch['image_description'] + sampled_batch['background_description']
            img = img.to(device)
            class_text = f"an image of {sampled_batch['class']}"

            text_features = loaded_model.encode_text(text_description)
            class_features = loaded_model.encode_text(class_text)

            # normalized features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            class_features = class_features / class_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = logit_scale.exp()
            logits_per_text_description = logit_scale * text_features @ class_features.t()

            if binary:
                selected_logits = logits_per_text_description[:, 0].unsqueeze(1)  # Now shape: [32, 1] as in training
                preds = torch.sigmoid(selected_logits).detach().cpu().numpy()
                probs = (preds > 0.5).astype(int)
            else:
                probs = logits_per_text_description.softmax(dim=-1).detach().cpu().numpy()

            img_name.extend(name)
            gt_labels.extend(label)
            scores.extend(probs)

    return img_name, gt_labels, scores

def main():
    text = ["waterbird", "landbird"]
    dataset = "waterbirds"
    scratch_dir = os.getenv("SCRATCH")
    img_dir = scratch_dir + '/datasets'
    model_type = "ViT-B/32"
    save_path = scratch_dir + '/results'

    model, preprocess, text_inputs, device = load_model(model_type, text)
    read_dataset = get_organized_dataset(base_dataset_path=Path(img_dir), dataset_name=dataset, dataset_split='test')
    loader_dataset = CLIPDataloader(clip_transform= preprocess, learning_data= read_dataset)
    loader = torch.utils.data.DataLoader(loader_dataset, batch_size=32, shuffle=False)

    img_name, gt_labels, scores = clip_inference_text(model, text_inputs, loader, device)

