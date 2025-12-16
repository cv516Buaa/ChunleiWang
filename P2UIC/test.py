import random
from data import ImageDetectionsField, TextField, RawField
from data import Sydney, UCM, RSICD, DataLoader,UIC, SUIM_IC, UWSeg_IC
from models.transformer.mamba_lm import MambaLM, MambaLMConfig
import evaluation.evaluation
from models.transformer import Transformer, VisualEncoder, MeshedDecoder, ScaledDotProductAttention
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import os
import warnings
import json
warnings.filterwarnings("ignore")


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_torch()


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    #total_params = sum(p.numel() for p in model.parameters())
    #print(f"total_params: {total_params}")
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.evaluation.compute_scores(gts, gen)



    return scores

if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Mamba-UIC')
    parser.add_argument('--exp_name', type=str, default='UWSeg_IC')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)

    #parser = argparse.ArgumentParser(description='MG-Transformer')
    #parser.add_argument('--exp_name', type=str, default='UIC') 
    #parser.add_argument('--batch_size', type=int, default=50)
    #parser.add_argument('--workers', type=int, default=0)

    parser.add_argument('--annotation_folder', type=str,
                        default='./dataset/UWSeg_IC')
    parser.add_argument('--features_path', type=str,
                        default='./clip_feature/UWSeg_IC_224')

    args = parser.parse_args()

    print('Evaluation')

    image_field = ImageDetectionsField(detections_path=args.features_path)

    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    if args.exp_name == 'UIC':
        dataset = UIC(image_field, text_field, 'UIC/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'UWSeg_IC':
        dataset = UWSeg_IC(image_field, text_field, 'UWSeg_IC/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'SUIM_IC':
        dataset = SUIM_IC(image_field, text_field, 'SUIM_IC/images/', args.annotation_folder, args.annotation_folder)

    _, _, test_dataset = dataset.splits

    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))


    
    encoder = VisualEncoder(3, 0, attention_module=ScaledDotProductAttention)
    decoder = MambaLM(lm_config=MambaLMConfig, vocab_size=len(text_field.vocab), max_len=127,
                      padding_idx=text_field.vocab.stoi['<pad>'])

    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

   
    data = torch.load('./saved_models/UWSeg_IC_best.pth')

    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)
    print("Test scores", scores)
    print('Sm', (scores['BLEU'][3]+scores['METEOR']+scores['ROUGE']+scores['CIDEr'])/4)
    '''
    print(scores['BLEU'][0])
    print(scores['BLEU'][1])
    print(scores['BLEU'][2])
    print(scores['BLEU'][3])
    print(scores['METEOR'])
    print(scores['ROUGE'])
    print(scores['CIDEr'])
    print(scores['SPICE'])
    '''
