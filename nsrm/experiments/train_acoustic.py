import torch
import torch.optim as optim
import scipy.io.wavfile
import numpy as np
from nsrm.experts.acoustic import ManifoldAcoustic

def train_sound_generation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ManifoldAcoustic(latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Concept: "Sound"
    code_sound = torch.randn(1, 16).to(device)
    
    # Target: 440Hz Sine Wave (Duration 1 sec, Sample Rate 16000)
    sr = 16000
    t_vals = torch.linspace(0, 1, sr).view(1, sr, 1).to(device)
    
    # Target Formula: sin(2 * pi * frequency * time)
    target_wave = torch.sin(2 * np.pi * 440 * t_vals)
    
    print("Starting Acoustic Training (Target: 440Hz Tone)...")
    
    for epoch in range(501):
        # We sample random chunks of time to train efficiently
        # (Training on the whole second every time is slow)
        idx = torch.randint(0, sr, (1, 1024)).to(device)
        t_batch = t_vals[:, idx.squeeze(), :]
        target_batch = target_wave[:, idx.squeeze(), :]
        
        # Forward Pass
        pred_wave = model(t_batch, code_sound)
        
        # Loss (MSE)
        loss = torch.nn.functional.mse_loss(pred_wave, target_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.5f}")

    # --- Verification: Generate Full Audio ---
    print("Generating full audio file...")
    with torch.no_grad():
        full_audio = model(t_vals, code_sound).cpu().numpy().squeeze()
    
    # Save to WAV
    output_filename = "nsrm_tone.wav"
    scipy.io.wavfile.write(output_filename, sr, full_audio)
    print(f"Saved '{output_filename}'. Play this file to hear the AI.")

if __name__ == "__main__":
    train_sound_generation()
