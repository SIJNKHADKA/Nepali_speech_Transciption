import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa
import time
import os
from tqdm import tqdm
import edit_distance as ed
import matplotlib.pyplot as plt
import json
from datetime import datetime


from model.configs import SR, device_name, UNQ_CHARS, INPUT_DIM, MODEL_NAME, NUM_UNQ_CHARS
from model.utils import CER_from_mfccs, batchify, clean_single_wav, gen_mfcc, indices_from_texts, load_model
from model.model import get_model


def train_model(model, optimizer, train_wavs, train_texts, test_wavs, test_texts, epochs=100, batch_size=50):
    """
    Train the model and track all metrics
    """
    # Lists to store history
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_cer': [],
        'epoch_time': []
    }
    
    best_cer = float('inf')
    best_epoch = 0

    with tf.device(device_name):

        for e in range(0, epochs):
            start_time = time.time()

            len_train = len(train_wavs)
            len_test = len(test_wavs)
            train_loss = 0
            test_loss = 0
            test_CER = 0
            train_batch_count = 0
            test_batch_count = 0

            print("\n" + "="*60)
            print(f"EPOCH {e+1}/{epochs}")
            print("="*60)
            
            # Training phase
            print("Training phase:")
            for start in tqdm(range(0, len_train, batch_size), desc="Training"):

                end = min(start + batch_size, len_train)
                
                x, target, target_lengths, output_lengths = batchify(
                    train_wavs[start:end], train_texts[start:end], UNQ_CHARS)

                with tf.GradientTape() as tape:
                    output = model(x, training=True)

                    loss = K.ctc_batch_cost(
                        target, output, output_lengths, target_lengths)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                train_loss += np.average(loss.numpy())
                train_batch_count += 1

            # Testing/Validation phase
            print("\nValidation phase:")
            for start in tqdm(range(0, len_test, batch_size), desc="Validating"):

                end = min(start + batch_size, len_test)
                
                x, target, target_lengths, output_lengths = batchify(
                    test_wavs[start:end], test_texts[start:end], UNQ_CHARS)

                output = model(x, training=False)

                # Calculate CTC Loss
                loss = K.ctc_batch_cost(
                    target, output, output_lengths, target_lengths)

                test_loss += np.average(loss.numpy())
                test_batch_count += 1

                # Computing CER metric
                input_len = np.ones(output.shape[0]) * output.shape[1]
                decoded_indices = K.ctc_decode(output, input_length=input_len,
                                       greedy=False, beam_width=100)[0][0]
                
                target_indices = [sent[sent != 0].tolist() for sent in target]
                predicted_indices = [sent[sent > 1].numpy().tolist() for sent in decoded_indices]

                len_batch = end - start
                for i in range(len_batch):
                    pred = predicted_indices[i]
                    truth = target_indices[i]
                    if len(truth) > 0:
                        sm = ed.SequenceMatcher(pred, truth)
                        ed_dist = sm.distance()
                        test_CER += ed_dist / len(truth)
                    
            # Calculate averages
            train_loss /= train_batch_count
            test_loss /= test_batch_count
            test_CER = (test_CER / len_test) * 100  # Convert to percentage
            epoch_time = time.time() - start_time

            # Store in history
            history['train_loss'].append(float(train_loss))
            history['test_loss'].append(float(test_loss))
            history['test_cer'].append(float(test_CER))
            history['epoch_time'].append(float(epoch_time))

            # Track best model
            if test_CER < best_cer:
                best_cer = test_CER
                best_epoch = e + 1
                # Save best model
                model.save("model/best_model.h5")
                print(f"\n🌟 New best model saved! CER: {best_cer:.2f}%")

            # Print epoch results
            print("\n" + "-"*60)
            print(f"Epoch: {e+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {test_loss:.4f}")
            print(f"Validation CER: {test_CER:.2f}%")
            print(f"Time: {epoch_time:.2f} seconds")
            print(f"Best CER so far: {best_cer:.2f}% (Epoch {best_epoch})")
            print("-"*60)

            # Save checkpoint every 10 epochs
            if (e + 1) % 10 == 0:
                model.save(f"model/checkpoint_epoch_{e+1}.h5")
                print(f"✅ Checkpoint saved at epoch {e+1}")

    return history, best_cer, best_epoch


def plot_training_curves(history, save_dir="model"):
    """
    Plot and save training curves
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Performance Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['test_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation CER
    axes[0, 1].plot(epochs, history['test_cer'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('CER (%)', fontsize=12)
    axes[0, 1].set_title('Validation Character Error Rate (CER)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    min_cer = min(history['test_cer'])
    min_epoch = history['test_cer'].index(min_cer) + 1
    axes[0, 1].axhline(y=min_cer, color='r', linestyle='--', label=f'Best CER: {min_cer:.2f}%')
    axes[0, 1].legend(fontsize=10)
    
    # Plot 3: Epoch Time
    axes[1, 0].plot(epochs, history['epoch_time'], 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss Difference (Overfitting indicator)
    loss_diff = [test - train for test, train in zip(history['test_loss'], history['train_loss'])]
    axes[1, 1].plot(epochs, loss_diff, 'orange', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss Difference', fontsize=12)
    axes[1, 1].set_title('Validation - Training Loss (Overfitting Indicator)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: {os.path.join(save_dir, 'training_curves.png')}")
    plt.close()


def generate_performance_report(history, best_cer, best_epoch, train_samples, test_samples, 
                                 total_time, config_params, save_dir="model"):
    """
    Generate comprehensive performance report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate additional metrics
    final_train_loss = history['train_loss'][-1]
    final_test_loss = history['test_loss'][-1]
    final_cer = history['test_cer'][-1]
    avg_epoch_time = np.mean(history['epoch_time'])
    
    # Improvement metrics
    initial_cer = history['test_cer'][0]
    cer_improvement = initial_cer - best_cer
    cer_improvement_pct = (cer_improvement / initial_cer) * 100
    
    # Generate report
    report = f"""
{'='*80}
                    SPEECH RECOGNITION MODEL TRAINING REPORT
{'='*80}

TRAINING CONFIGURATION
{'-'*80}
Model Name:              {config_params.get('model_name', 'N/A')}
Total Epochs:            {len(history['train_loss'])}
Batch Size:              {config_params.get('batch_size', 'N/A')}
Training Samples:        {train_samples}
Validation Samples:      {test_samples}
Sample Rate:             {config_params.get('sr', 'N/A')} Hz
Total Training Time:     {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)
Average Time per Epoch:  {avg_epoch_time:.2f} seconds

MODEL ARCHITECTURE
{'-'*80}
Input Dimension:         {config_params.get('input_dim', 'N/A')}
Number of Characters:    {config_params.get('num_chars', 'N/A')}
RNN Type:                {config_params.get('rnn_type', 'N/A')}
RNN Layers:              {config_params.get('rnn_layers', 'N/A')}
RNN Dimension:           {config_params.get('rnn_dim', 'N/A')}
Bidirectional:           {config_params.get('bidirectional', 'N/A')}
CNN Layers:              {config_params.get('cnn_layers', 'N/A')}
CNN Filters:             {config_params.get('cnn_filters', 'N/A')}

KEY PERFORMANCE INDICATORS (KPIs)
{'='*80}

📊 LOSS METRICS
{'-'*80}
Initial Training Loss:       {history['train_loss'][0]:.4f}
Final Training Loss:         {final_train_loss:.4f}
Training Loss Improvement:   {history['train_loss'][0] - final_train_loss:.4f}

Initial Validation Loss:     {history['test_loss'][0]:.4f}
Final Validation Loss:       {final_test_loss:.4f}
Validation Loss Improvement: {history['test_loss'][0] - final_test_loss:.4f}

🎯 CHARACTER ERROR RATE (CER)
{'-'*80}
Initial CER:                 {initial_cer:.2f}%
Final CER:                   {final_cer:.2f}%
Best CER:                    {best_cer:.2f}%
Best CER Epoch:              {best_epoch}
CER Improvement:             {cer_improvement:.2f}% points
CER Improvement Percentage:  {cer_improvement_pct:.2f}%

📈 MODEL CONVERGENCE
{'-'*80}
Epochs to Best Model:        {best_epoch}
Convergence Efficiency:      {(best_epoch / len(history['train_loss'])) * 100:.1f}%

🔍 OVERFITTING ANALYSIS
{'-'*80}
Final Loss Gap (Val - Train): {final_test_loss - final_train_loss:.4f}
"""
    
    # Add overfitting assessment
    loss_gap = final_test_loss - final_train_loss
    if loss_gap < 0.5:
        report += "Assessment:                   ✅ Good generalization\n"
    elif loss_gap < 1.0:
        report += "Assessment:                   ⚠️  Slight overfitting\n"
    else:
        report += "Assessment:                   ❌ Significant overfitting detected\n"
    
    report += f"""
⏱️  EFFICIENCY METRICS
{'-'*80}
Total Training Time:         {total_time/3600:.2f} hours
Avg Time per Epoch:          {avg_epoch_time:.2f} seconds
Samples Processed:           {train_samples * len(history['train_loss']):,}
Training Throughput:         {(train_samples * len(history['train_loss'])) / total_time:.2f} samples/second

📝 FINAL SUMMARY
{'-'*80}
✓ Model successfully trained for {len(history['train_loss'])} epochs
✓ Best model achieved {best_cer:.2f}% CER at epoch {best_epoch}
✓ Overall CER improvement: {cer_improvement_pct:.2f}%
✓ Model saved at: model/best_model.h5
✓ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
"""
    
    # Save report to file
    report_path = os.path.join(save_dir, 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save metrics as JSON for later analysis
    metrics_data = {
        'config': config_params,
        'history': history,
        'best_cer': float(best_cer),
        'best_epoch': int(best_epoch),
        'final_metrics': {
            'train_loss': float(final_train_loss),
            'test_loss': float(final_test_loss),
            'cer': float(final_cer)
        },
        'training_info': {
            'train_samples': train_samples,
            'test_samples': test_samples,
            'total_time_seconds': float(total_time),
            'avg_epoch_time': float(avg_epoch_time)
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    json_path = os.path.join(save_dir, 'training_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(report)
    print(f"\n Detailed report saved to: {report_path}")
    print(f" Metrics data saved to: {json_path}")
    
    return report


def load_data(base_dir, csv_filename="metadata.csv"):
    """
    Load audio files and transcriptions from the specified directory structure
    """
    output_dir = os.path.join(base_dir, "nepali_merged")
    csv_path = os.path.join(output_dir, csv_filename)
    
    print(f"Reading CSV from: {csv_path}")
    texts_df = pd.read_csv(csv_path)
    
    print(f"CSV columns: {texts_df.columns.tolist()}")
    print(f"Total samples in CSV: {len(texts_df)}")
    
    # Auto-detect columns
    file_col = None
    text_col = None
    
    for col in texts_df.columns:
        col_lower = col.lower()
        if 'file' in col_lower or 'path' in col_lower or 'audio' in col_lower:
            file_col = col
        if 'text' in col_lower or 'transcript' in col_lower or 'sentence' in col_lower:
            text_col = col
    
    if file_col is None or text_col is None:
        print("Could not automatically detect columns. Available columns:")
        print(texts_df.columns.tolist())
        file_col = texts_df.columns[0]
        text_col = texts_df.columns[1]
        print(f"Using: file_col='{file_col}', text_col='{text_col}'")
    
    train_wavs = []
    train_texts = []
    failed_files = []
    
    print(f"Loading audio files from: {output_dir}")
    
    for idx, row in tqdm(texts_df.iterrows(), total=len(texts_df), desc="Loading audio"):
        try:
            file_name = row[file_col]
            file_name_base = os.path.splitext(file_name)[0]
            audio_path = os.path.join(output_dir, f"{file_name_base}.flac")
            
            if not os.path.exists(audio_path):
                audio_path = os.path.join(output_dir, file_name)
            
            if os.path.exists(audio_path):
                wav, _ = librosa.load(audio_path, sr=SR)
                train_wavs.append(wav)
                train_texts.append(row[text_col])
            else:
                failed_files.append(file_name)
                
        except Exception as e:
            print(f"\nError loading {file_name}: {str(e)}")
            failed_files.append(file_name)
    
    print(f"\nSuccessfully loaded: {len(train_wavs)} audio files")
    if failed_files:
        print(f"Failed to load: {len(failed_files)} files")
        print(f"First few failed files: {failed_files[:5]}")
    
    return train_wavs, train_texts


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("    SPEECH RECOGNITION MODEL TRAINING - NEPALI DATASET")
    print("="*80 + "\n")

    # Dataset paths
    BASE_DIR = r"E:\DATASET"
    
    # Training configuration
    EPOCHS = 100
    BATCH_SIZE = 2
    TEST_SIZE = 0.2
    
    training_start_time = time.time()
    
    # Definition of the model
    print(" Initializing model...")
    model = get_model(INPUT_DIM, NUM_UNQ_CHARS, num_res_blocks=5, num_cnn_layers=2,
                      cnn_filters=50, cnn_kernel_size=15, rnn_dim=170, rnn_dropout=0.15, 
                      num_rnn_layers=2, num_dense_layers=1, dense_dim=340, 
                      model_name=MODEL_NAME, rnn_type="lstm", use_birnn=True)
    print(" Model initialized successfully\n")

    # Definition of the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Load the data
    print(" Loading data...")
    train_wavs, train_texts = load_data(base_dir=BASE_DIR, csv_filename="metadata.csv")
    print(" Data loaded successfully\n")
    
    # Clean the audio files
    print("Cleaning audio files...")
    train_wavs = [clean_single_wav(wav) for wav in tqdm(train_wavs, desc="Cleaning")]
    print(" Audio files cleaned\n")

    # Generate MFCC features
    print(" Generating MFCC features...")
    train_wavs = [gen_mfcc(wav) for wav in tqdm(train_wavs, desc="MFCC")]
    print(" MFCC features generated\n")

    # Train-Test Split
    print("  Splitting data...")
    train_wavs, test_wavs, train_texts, test_texts = train_test_split(
        train_wavs, train_texts, test_size=TEST_SIZE, random_state=42)
    
    print(f"Training samples: {len(train_wavs)}")
    print(f"Validation samples: {len(test_wavs)}\n")

    # Store configuration for report
    config_params = {
        'model_name': MODEL_NAME,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'sr': SR,
        'input_dim': INPUT_DIM,
        'num_chars': NUM_UNQ_CHARS,
        'rnn_type': 'LSTM',
        'rnn_layers': 2,
        'rnn_dim': 170,
        'bidirectional': True,
        'cnn_layers': 2,
        'cnn_filters': 50
    }

    # Train the model
    print(" Starting training...")
    history, best_cer, best_epoch = train_model(
        model, optimizer, train_wavs, train_texts,
        test_wavs, test_texts, epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    total_training_time = time.time() - training_start_time

    # Save the final model
    print("\n Saving final model...")
    os.makedirs("model", exist_ok=True)
    model.save("model/trained_model_final2.h5")
    print("Final model saved\n")

    # Generate training curves
    print(" Generating training curves...")
    plot_training_curves(history, save_dir="model")

    # Generate comprehensive performance report
    print("\n Generating performance report...")
    generate_performance_report(
        history, best_cer, best_epoch,
        len(train_wavs), len(test_wavs),
        total_training_time, config_params,
        save_dir="model"
    )

    print("\n" + "="*80)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll outputs saved in 'model/' directory:")
    print("   - best_model.h5 (Best performing model)")
    print("   - trained_model_final.h5 (Final model)")
    print("   - training_curves.png (Visualization)")
    print("   - training_report.txt (Detailed report)")
    print("   - training_metrics.json (Metrics data)")
    print("\n")