import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from net import SoundStream, WaveDiscriminator, STFTDiscriminator
from dataset import *

import pdb
import torchaudio
import random
import os
import argparse
import sys
import matplotlib.pyplot as plt

def collate_fn(batch):
    length=48000
    def process_element(elem, length):
        if elem.shape[-1] > length:
            start_index = random.randint(0, elem.shape[-1] - length)
            return elem[:, start_index:start_index + length]
        else:
            padding = (0, length - elem.shape[-1])
            return F.pad(elem, padding, 'constant', 0)
    
    processed_batch = [process_element(elem, length) for elem in batch]
    padded_batch = torch.stack(processed_batch, dim=0)
    lengths = torch.full((len(batch),), length, dtype=torch.long)
    
    return padded_batch, lengths

def adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()
    
    stft_loss = F.relu(1-features_stft_disc_G_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    wave_loss = torch.cat([F.relu(1-features_wave_disc_G_x[key][-1]).sum(dim=2).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names])
    loss = torch.cat([stft_loss, wave_loss]).mean()
    
    return loss

def feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft):
    wave_disc_names = lengths_wave.keys()
    
    stft_loss = torch.stack([((feat_x-feat_G_x).abs().sum(dim=-1)/lengths_stft[i].view(-1,1,1)).sum(dim=-1).sum(dim=-1) for i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))], dim=1).mean(dim=1, keepdim=True)
    wave_loss = torch.stack([torch.stack([(feat_x-feat_G_x).abs().sum(dim=-1).sum(dim=-1)/lengths_wave[key][i] for i, (feat_x, feat_G_x) in enumerate(zip(features_wave_disc_x[key], features_wave_disc_G_x[key]))], dim=1) for key in wave_disc_names], dim=2).mean(dim=1)
    loss = torch.cat([stft_loss, wave_loss], dim=1).mean()
    
    return loss

def spectral_reconstruction_loss(x, G_x, eps=1e-4):
    L = 0
    for i in range(6,12):
        s = 2**i
        alpha_s = (s/2)**0.5
        melspec = MelSpectrogram(sample_rate=sr, n_fft=s, hop_length=s//4, n_mels=8, wkwargs={"device": device}).to(device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        
        loss = (S_x-S_G_x).abs().sum() + alpha_s*(((torch.log(S_x.abs()+eps)-torch.log(S_G_x.abs()+eps))**2).sum(dim=-2)**0.5).sum()
        L += loss
    
    return L

def adversarial_d_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()
    
    real_stft_loss = F.relu(1-features_stft_disc_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    real_wave_loss = torch.stack([F.relu(1-features_wave_disc_x[key][-1]).sum(dim=-1).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names], dim=1)
    real_loss = torch.cat([real_stft_loss.view(-1,1), real_wave_loss], dim=1).mean()
    
    generated_stft_loss = F.relu(1+features_stft_disc_G_x[-1]).sum(dim=-1).squeeze()/lengths_stft[-1].squeeze()
    generated_wave_loss = torch.stack([F.relu(1+features_wave_disc_G_x[key][-1]).sum(dim=-1).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names], dim=1)
    generated_loss = torch.cat([generated_stft_loss.view(-1,1), generated_wave_loss], dim=1).mean()
    
    return real_loss + generated_loss

def ensure_path_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def show_loss(history):
    log_file_path = "./loss/loss_history.log"
    ensure_path_exists(log_file_path)

    with open(log_file_path, 'w') as log_file:
        history_json = json.dumps(history, indent=4)
        log_file.write(history_json)

    plt.figure(figsize=(10, 5))

    plt.plot(history["train"]["d"], label='Train Discriminator Loss')
    plt.plot(history["train"]["g"], label='Train Generator Loss')
    plt.plot(history["valid"]["d"], label='Validation Discriminator Loss')
    plt.plot(history["valid"]["g"], label='Validation Generator Loss')
    plt.plot(history["valid"]["both"], label='Validation Total Loss')
    plt.plot(history["test"]["d"], label='Test Discriminator Loss')
    plt.plot(history["test"]["g"], label='Test Generator Loss')

    plt.legend()
    plt.title('Losses over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    save_path = "./loss/losses_over_epochs.png"
    ensure_path_exists(save_path)
    plt.savefig(save_path)

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="process parameters")
    parser.add_argument('-srvq', action='store_true', help='use self residual quantizer')
    args = parser.parse_args()

    use_srvq = False
    if args.srvq:
        use_srvq = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LAMBDA_ADV = 1
    LAMBDA_FEAT = 100
    LAMBDA_REC = 1
    N_EPOCHS = 15
    BATCH_SIZE = 32 # 4

    soundstream = SoundStream(C=1, D=150, n_q=1, codebook_size=1, use_srvq=use_srvq)
    wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
    W, H = 1024, 256
    stft_disc = STFTDiscriminator(C=1, F_bins=W//2)

    soundstream.to(device)
    wave_disc.to(device)
    stft_disc.to(device)

    torchaudio.set_audio_backend("soundfile")

    # train_dataset = NSynthDataset(audio_dir="./data/nsynth-train.jsonwav/nsynth-train/audio")
    train_dataset = JsonDataset(json_dir="/home3/hexin/speechbrain/recipes/VoxLingua107/lang_id/fleurs_train.json")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=1)
    sr = train_dataset.sr

    # valid_dataset = NSynthDataset(audio_dir="./data/nsynth-valid.jsonwav/nsynth-valid/audio")
    valid_dataset = JsonDataset(json_dir="/home3/hexin/speechbrain/recipes/VoxLingua107/lang_id/fleurs_valid.json")
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=1)

    # test_dataset = NSynthDataset(audio_dir="./data/nsynth-test.jsonwav/nsynth-test/audio")
    test_dataset = JsonDataset(json_dir="/home3/hexin/speechbrain/recipes/VoxLingua107/lang_id/fleurs_test.json")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=1)

    optimizer_g = optim.Adam(soundstream.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))

    criterion_g = lambda x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft: LAMBDA_ADV*adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave) + LAMBDA_FEAT*feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft) + LAMBDA_REC*spectral_reconstruction_loss(x, G_x)
    criterion_d = adversarial_d_loss

    best_model = soundstream.state_dict().copy()
    best_val_loss_g = float("inf")
    best_val_loss_d = float("inf")
    best_val_loss_both = float("inf")

    early_stop = False

    history = {
        "train": {"d": [], "g": []},
        "valid": {"d": [], "g": [], "both": []},
        "test": {"d": [], "g": []}
    }

    for epoch in range(1, N_EPOCHS+1):
        
        soundstream.train()
        stft_disc.train()
        wave_disc.train()
        
        train_loss_d = 0.0
        train_loss_g = 0.0
        print("process train")
        for x, lengths_x in tqdm(train_loader):
            x = x.to(device)
            lengths_x = lengths_x.to(device)
            
            G_x = soundstream(x)
            
            s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
            lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
            s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
            
            lengths_stft = stft_disc.features_lengths(lengths_s_x)
            lengths_wave = wave_disc.features_lengths(lengths_x)
            
            features_stft_disc_x = stft_disc(s_x)
            features_wave_disc_x = wave_disc(x)
            
            features_stft_disc_G_x = stft_disc(s_G_x)
            features_wave_disc_G_x = wave_disc(G_x)
            
            loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
            train_loss_g += loss_g.item()
            
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            
            features_stft_disc_x = stft_disc(s_x)
            features_wave_disc_x = wave_disc(x)
            
            features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
            features_wave_disc_G_x_det = wave_disc(G_x.detach())
            
            loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
            
            train_loss_d += loss_d.item()
            
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
        
        history["train"]["d"].append(train_loss_d/len(train_loader))
        history["train"]["g"].append(train_loss_g/len(train_loader))
        
        with torch.no_grad():
            stft_disc.eval()
            wave_disc.eval()
            
            valid_loss_d = 0.0
            valid_loss_g = 0.0
            print("process valid")
            for x, lengths_x in tqdm(valid_loader):
                x = x.to(device)
                lengths_x = lengths_x.to(device)
            
                G_x = soundstream(x)
                
                s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
                lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
                s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
                
                lengths_stft = stft_disc.features_lengths(lengths_s_x)
                lengths_wave = wave_disc.features_lengths(lengths_x)
                
                features_stft_disc_x = stft_disc(s_x)
                features_wave_disc_x = wave_disc(x)
                
                features_stft_disc_G_x = stft_disc(s_G_x)
                features_wave_disc_G_x = wave_disc(G_x)
                
                loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
                valid_loss_g += loss_g.item()
                
                features_stft_disc_x = stft_disc(s_x)
                features_wave_disc_x = wave_disc(x)
                
                features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
                features_wave_disc_G_x_det = wave_disc(G_x.detach())
                
                loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
                
                valid_loss_d += loss_d.item()
            
            history["valid"]["d"].append(valid_loss_d/len(valid_loader))
            history["valid"]["g"].append(valid_loss_g/len(valid_loader))
            history["valid"]["both"].append(valid_loss_d/len(valid_loader) + valid_loss_g/len(valid_loader))

            if best_val_loss_d > history["valid"]["d"][-1]:
                best_model = soundstream.state_dict().copy()
                save_path = f'./model/best_model_d/best_model_d_{epoch}.pth'
                ensure_path_exists(save_path)
                torch.save(best_model, save_path)
        
            if best_val_loss_g > history["valid"]["g"][-1]:
                best_model = soundstream.state_dict().copy()
                save_path = f'./model/best_model_g/best_model_g_{epoch}.pth'
                ensure_path_exists(save_path)
                torch.save(best_model, save_path)

            if best_val_loss_both > history["valid"]["both"][-1]:
                best_model = soundstream.state_dict().copy()
                save_path = f'./model/best_model_both/best_model_both_{epoch}.pth'
                ensure_path_exists(save_path)
                torch.save(best_model, save_path)

            if len(history["valid"]["both"]) > 2:
                loss_diff_both_1 = history["valid"]["both"][-2] - history["valid"]["both"][-1]
                loss_diff_both_2 = history["valid"]["both"][-3] - history["valid"]["both"][-2]
                if loss_diff_both_1 < 0 or loss_diff_both_2 < 0:
                    early_stop = True
                elif abs(loss_diff_both_2-loss_diff_both_1) < 0.05:
                    early_stop = True
        
        with torch.no_grad():
            stft_disc.eval()
            wave_disc.eval()
            
            test_loss_d = 0.0
            test_loss_g = 0.0
            print("process test")
            for x, lengths_x in tqdm(test_loader):
                x = x.to(device)
                lengths_x = lengths_x.to(device)
            
                G_x = soundstream(x)
                
                s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
                lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
                s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
                
                lengths_stft = stft_disc.features_lengths(lengths_s_x)
                lengths_wave = wave_disc.features_lengths(lengths_x)
                
                features_stft_disc_x = stft_disc(s_x)
                features_wave_disc_x = wave_disc(x)
                
                features_stft_disc_G_x = stft_disc(s_G_x)
                features_wave_disc_G_x = wave_disc(G_x)
                
                loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
                test_loss_g += loss_g.item()
                
                features_stft_disc_x = stft_disc(s_x)
                features_wave_disc_x = wave_disc(x)
                
                features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
                features_wave_disc_G_x_det = wave_disc(G_x.detach())
                
                loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
                
                test_loss_d += loss_d.item()
            
            history["test"]["d"].append(test_loss_d/len(test_loader))
            history["test"]["g"].append(test_loss_g/len(test_loader))
                
        if early_stop:
            break
    
    show_loss(history)
