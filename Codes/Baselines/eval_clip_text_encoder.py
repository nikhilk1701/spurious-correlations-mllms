import json
import clip
from torch import nn

from eval_clip import *

def test_clip_text_enc(loaded_model):
    siamese_cat_enc = loaded_model.encode_text(clip.tokenize("black cat"))
    persian_cat_enc = loaded_model.encode_text(clip.tokenize("orange cat"))
    not_cat_enc = loaded_model.encode_text(clip.tokenize("not cat"))

    encs = torch.stack([siamese_cat_enc, persian_cat_enc, not_cat_enc], dim=0).squeeze(dim=1)
    print(encs.norm(dim=1, keepdim=True))
    encs = encs/encs.norm(dim=1, keepdim=True)
    print(encs.size())
    
    siamese_cat_enc = siamese_cat_enc / siamese_cat_enc.norm(dim=1, keepdim=True)
    persian_cat_enc = persian_cat_enc / persian_cat_enc.norm(dim=1, keepdim=True)
    not_cat_enc = not_cat_enc / not_cat_enc.norm(dim=1, keepdim=True)

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * encs @ encs.t()

    # print(logits[:, 1:])
    # print(logits[:, 1:].softmax(dim=1))
    # print(logits[:, [0, 2]].softmax(dim=1))
    # print(logits[:, [0, 1]].softmax(dim=1))

    # print("s x s", logit_scale * siamese_cat_enc @ siamese_cat_enc.t())
    # print("p x p", logit_scale * persian_cat_enc @ persian_cat_enc.t())
    # print("n x n", logit_scale * not_cat_enc @ not_cat_enc.t())

    # print("s x p", logit_scale * siamese_cat_enc @ persian_cat_enc.t())
    # print("n x p", logit_scale * not_cat_enc @ persian_cat_enc.t())
    # print("n x s", logit_scale * not_cat_enc @ siamese_cat_enc.t())

    sp= logit_scale * siamese_cat_enc @ persian_cat_enc.t()
    nps= logit_scale * not_cat_enc @ persian_cat_enc.t()
    ns= logit_scale * not_cat_enc @ siamese_cat_enc.t()

    concat = torch.cat((sp, ns), dim=0)
    probs = concat.softmax(dim=0)
    print(concat)
    print(probs)



    

def clip_inference_text(loaded_model, text_inputs, device, binary=False):
    waterbird_ctext_enc = loaded_model.encode_text(clip.tokenize(f"an image of waterbird"))
    landbird_ctext_enc = loaded_model.encode_text(clip.tokenize(f"an image of landbird"))
    waterbird_ctext_enc = waterbird_ctext_enc / waterbird_ctext_enc.norm(dim=1, keepdim=True)
    landbird_ctext_enc = landbird_ctext_enc / landbird_ctext_enc.norm(dim=1, keepdim=True)

    cnt_correct = 0.0

    with torch.no_grad():
        for ip in text_inputs:
            # print(ip)
            label = ip["label"]
            text_description = clip.tokenize(ip['text'], 77, True)
            text_features = loaded_model.encode_text(text_description)

            # normalized features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
            logits_waterbird = logit_scale * text_features @ waterbird_ctext_enc.t()
            logits_landbird = logit_scale * text_features @ landbird_ctext_enc.t()

            if label == 'waterbird' and logits_waterbird > logits_landbird or label == 'landbird' and logits_landbird > logits_waterbird:
                cnt_correct += 1.0

    accuracy = cnt_correct * 1.0 / len(text_inputs)
    print("accuracy = ", accuracy)

def main():
    text = ["waterbird", "landbird"]
    model_type = "ViT-B/32"

    model, _, text_inputs, device = load_model(model_type, text)

    text_inputs = []
    with open("/scratch/nk3853/test_2024-11-07-8-24/answers_object.jsonl", 'r') as afile, open("./outputs/questions_object_test.jsonl") as qfile:
        for qline, aline in zip(qfile,afile):
            ques = json.loads(qline)
            ans = json.loads(aline)
            ans["label"] = ques["label"]
            text_inputs.append(ans)

    clip_inference_text(model, text_inputs, device)

main()