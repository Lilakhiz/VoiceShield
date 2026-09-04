#!/usr/bin/env python3
"""
Replay Detector Calibration Utility

This script evaluates the signal-heuristic replay detector against
labeled evaluation data (e.g., ASVspoof2019 PA).

Usage:
    python calibrate_replay.py --data-dir /path/to/asvspoof2019_PA --output results.json

Expected dataset structure (ASVspoof2019 PA):
    data_dir/
        ASVspoof2019_PA_asv_protocols/
            ASVspoof2019.PA.cm.eval.trl.txt
        ASVspoof2019_PA_eval/
            *.flac

The protocol file format:
    SPEAKER_ID  FILE_ID  -  SYSTEM_ID  KEY
    where KEY is "bonafide" (genuine) or "spoof" (replay attack)

If ASVspoof data is not available, this script can also evaluate against
synthetic test signals (as used in test_signal_detectors.py).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import librosa
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.replay_detection import detect_replay


def load_asvspoof_protocol(protocol_path: Path) -> List[Tuple[str, str]]:
    """
    Load ASVspoof protocol file.
    Returns list of (file_id, label) where label is 'bonafide' or 'spoof'.
    """
    entries = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                speaker_id, file_id, _, system_id, key = parts[:5]
                entries.append((file_id, key))
    return entries


def evaluate_detector(audio_dir: Path, protocol_entries: List[Tuple[str, str]], 
                      sample_rate: int = 16000) -> dict:
    """
    Run detector on all audio files and compute metrics.
    """
    y_true = []
    y_scores = []
    results = []
    
    for file_id, label in protocol_entries:
        # Try multiple extensions
        audio_path = None
        for ext in ['.flac', '.wav', '.FLAC', '.WAV']:
            candidate = audio_dir / (file_id + ext)
            if candidate.exists():
                audio_path = candidate
                break
        
        if audio_path is None:
            print(f"Warning: Audio file not found for {file_id}")
            continue
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sample_rate)
            
            # Run detection
            result = detect_replay(y, sr)
            
            if not result.available:
                print(f"Warning: Detector unavailable for {file_id}: {result.error}")
                continue
            
            prob = result.replay_probability
            y_true.append(1 if label == 'spoof' else 0)
            y_scores.append(prob)
            results.append({
                'file_id': file_id,
                'label': label,
                'probability': prob,
                'method': result.method
            })
            
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            continue
    
    if not y_true:
        raise ValueError("No valid audio files processed")
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Compute metrics
    auc = roc_auc_score(y_true, y_scores)
    
    # Find optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Metrics at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    # Also compute at default 0.5 threshold
    y_pred_05 = (y_scores >= 0.5).astype(int)
    precision_05, recall_05, f1_05, _ = precision_recall_fscore_support(
        y_true, y_pred_05, average='binary'
    )
    
    metrics = {
        'roc_auc': float(auc),
        'optimal_threshold': float(optimal_threshold),
        'at_optimal_threshold': {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold': float(optimal_threshold)
        },
        'at_05_threshold': {
            'precision': float(precision_05),
            'recall': float(recall_05),
            'f1': float(f1_05),
            'threshold': 0.5
        },
        'num_samples': len(y_true),
        'num_bonafide': int(np.sum(y_true == 0)),
        'num_spoof': int(np.sum(y_true == 1)),
        'score_range': [float(np.min(y_scores)), float(np.max(y_scores))],
        'results': results
    }
    
    return metrics


def evaluate_synthetic() -> dict:
    """
    Evaluate on synthetic test signals (same as test_signal_detectors.py).
    This does NOT require ASVspoof data.
    """
    from app.services.replay_detection import detect_replay
    from app.services.deepfake_detection import _f0_jitter, _harmonic_to_noise_ratio, _formant_band_flatness
    from scipy.signal import butter, lfilter
    
    SR = 16000
    
    def _live_like_signal(duration=3.0, seed=0):
        t = np.linspace(0, duration, int(SR * duration))
        rng = np.random.default_rng(seed)
        f0_jitter = 150 + np.cumsum(rng.standard_normal(len(t)) * 0.5)
        phase = 2 * np.pi * np.cumsum(f0_jitter) / SR
        sig = (0.5 * np.sin(phase) + 0.15 * np.sin(3 * phase) + 0.05 * np.sin(7 * phase)
               + 0.03 * rng.standard_normal(len(t)))
        return sig.astype(np.float32)
    
    def _replay_like_signal(live_like):
        b, a = butter(4, 3000 / (SR / 2), btype="low")
        filtered = lfilter(b, a, live_like)
        return (np.tanh(filtered * 3) / 3).astype(np.float32)
    
    print("Generating synthetic test signals...")
    
    # Generate test signals
    n_trials = 50
    y_true = []
    y_scores = []
    
    for i in range(n_trials):
        # Genuine signal
        live = _live_like_signal(seed=i)
        result = detect_replay(live, SR)
        if result.available:
            y_true.append(0)
            y_scores.append(result.replay_probability)
        
        # Replay-like signal (low-pass filtered + compressed)
        replay = _replay_like_signal(live)
        result = detect_replay(replay, SR)
        if result.available:
            y_true.append(1)
            y_scores.append(result.replay_probability)
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    if len(y_true) == 0:
        raise ValueError("No valid detections")
    
    auc = roc_auc_score(y_true, y_scores)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_scores >= optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    return {
        'roc_auc': float(auc),
        'optimal_threshold': float(optimal_threshold),
        'at_optimal_threshold': {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold': float(optimal_threshold)
        },
        'num_samples': len(y_true),
        'note': 'Synthetic evaluation only - NOT calibrated on real ASVspoof PA data'
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate replay detector against labeled data"
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Path to ASVspoof2019 PA data directory'
    )
    parser.add_argument(
        '--protocol', type=str, default=None,
        help='Path to protocol file (default: ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.cm.eval.trl.txt under data-dir)'
    )
    parser.add_argument(
        '--audio-dir', type=str, default=None,
        help='Path to audio directory (default: ASVspoof2019_PA_eval under data-dir)'
    )
    parser.add_argument(
        '--output', type=str, default='calibration_results_replay.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Run synthetic evaluation only (no ASVspoof data needed)'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=16000,
        help='Sample rate for audio loading'
    )
    
    args = parser.parse_args()
    
    if args.synthetic:
        print("Running synthetic evaluation...")
        metrics = evaluate_synthetic()
    else:
        if not args.data_dir:
            parser.error("--data-dir required for ASVspoof evaluation (or use --synthetic)")
        
        data_dir = Path(args.data_dir)
        
        if args.protocol:
            protocol_path = Path(args.protocol)
        else:
            protocol_path = data_dir / 'ASVspoof2019_PA_asv_protocols' / 'ASVspoof2019.PA.cm.eval.trl.txt'
        
        if args.audio_dir:
            audio_dir = Path(args.audio_dir)
        else:
            audio_dir = data_dir / 'ASVspoof2019_PA_eval'
        
        if not protocol_path.exists():
            print(f"Error: Protocol file not found: {protocol_path}")
            sys.exit(1)
        
        if not audio_dir.exists():
            print(f"Error: Audio directory not found: {audio_dir}")
            sys.exit(1)
        
        print(f"Loading protocol from {protocol_path}...")
        protocol_entries = load_asvspoof_protocol(protocol_path)
        print(f"Found {len(protocol_entries)} entries")
        
        print(f"Evaluating detector on {audio_dir}...")
        metrics = evaluate_detector(audio_dir, protocol_entries, args.sample_rate)
    
    # Print summary
    print("\n=== Calibration Results ===")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Optimal threshold: {metrics['optimal_threshold']:.4f}")
    print(f"At optimal threshold:")
    print(f"  Precision: {metrics['at_optimal_threshold']['precision']:.4f}")
    print(f"  Recall:    {metrics['at_optimal_threshold']['recall']:.4f}")
    print(f"  F1:        {metrics['at_optimal_threshold']['f1']:.4f}")
    print(f"At 0.5 threshold:")
    print(f"  Precision: {metrics['at_05_threshold']['precision']:.4f}")
    print(f"  Recall:    {metrics['at_05_threshold']['recall']:.4f}")
    print(f"  F1:        {metrics['at_05_threshold']['f1']:.4f}")
    print(f"Samples: {metrics['num_samples']} (bonafide: {metrics.get('num_bonafide', 'N/A')}, spoof: {metrics.get('num_spoof', 'N/A')})")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Print recommended threshold
    print(f"\nRECOMMENDED: Set REPLAY_THRESHOLD={metrics['optimal_threshold']:.4f} in environment")
    print("NOTE: This threshold is for the signal-heuristic fallback only.")


if __name__ == '__main__':
    main()