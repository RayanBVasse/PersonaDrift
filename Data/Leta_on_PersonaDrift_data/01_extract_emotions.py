"""
LETA × PersonaDrift Integration — Script 1: Extract Emotions
=============================================================
Parses environment text files (WhatsApp + ChatGPT exports),
extracts Subject 533's messages, applies NRC Emotion Lexicon,
outputs per-message emotion scores as CSV.

Usage:
    python 01_extract_emotions.py --config config.json

Input:  Raw text exports (one .txt per environment)
Output: persona_drift_leta_emotions.csv

Requirements: pip install pandas numpy
"""

import re
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


# =============================================================================
# NRC LEXICON LOADER
# =============================================================================

def load_nrc_lexicon(nrc_path):
    """
    Load NRC Emotion Lexicon v0.92 (Wordlevel format).
    Format: word\temotion\t0|1 (tab-separated)
    Returns: dict mapping word -> set of emotions
    """
    lexicon = defaultdict(set)
    with open(nrc_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            word, emotion, value = parts
            if int(value) == 1:
                lexicon[word.lower()].add(emotion)
    print(f"[NRC] Loaded {len(lexicon)} words with emotion associations")
    return lexicon


# =============================================================================
# TEXT PARSERS
# =============================================================================

def parse_whatsapp_export(filepath, subject_identifiers):
    """
    Parse WhatsApp export format:
    DD/MM/YYYY, HH:MM - Speaker: Message

    Returns list of dicts: {timestamp, speaker, text, msg_index}
    """
    messages = []
    # Pattern: DD/MM/YYYY, HH:MM - Speaker: Message
    # Also handle DD/MM/YY format
    pattern = re.compile(
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(.+?):\s*(.*)'
    )

    current_msg = None
    msg_index = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            match = pattern.match(line)

            if match:
                # Save previous message if exists
                if current_msg:
                    messages.append(current_msg)

                date_str, time_str, speaker, text = match.groups()

                # Check if this is the subject
                speaker_clean = speaker.strip()
                is_subject = any(
                    sid.lower() in speaker_clean.lower()
                    for sid in subject_identifiers
                )

                current_msg = {
                    'timestamp': f"{date_str} {time_str}",
                    'speaker': speaker_clean,
                    'text': text.strip(),
                    'msg_index': msg_index,
                    'is_subject': is_subject
                }
                msg_index += 1
            else:
                # Continuation line — append to current message
                if current_msg:
                    current_msg['text'] += ' ' + line.strip()

    # Don't forget last message
    if current_msg:
        messages.append(current_msg)

    # Filter to subject only
    subject_msgs = [m for m in messages if m['is_subject']]
    print(f"[WhatsApp] Parsed {len(messages)} total messages, "
          f"{len(subject_msgs)} from subject")
    return subject_msgs


def parse_chatgpt_export(filepath, subject_tag='[Subj_533]'):
    """
    Parse ChatGPT export format:
    Lines starting with [Subj_533] or [ChatGPT] etc.
    Multi-line messages continue until next speaker tag.

    Returns list of dicts: {timestamp, speaker, text, msg_index}
    """
    messages = []
    # Pattern: [speaker_tag] text...
    # Also handle [user] tag
    tag_pattern = re.compile(r'^\[([^\]]+)\]\s*(.*)')

    current_msg = None
    msg_index = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            match = tag_pattern.match(line)

            if match:
                # Save previous message
                if current_msg:
                    messages.append(current_msg)

                speaker, text = match.groups()
                speaker_clean = speaker.strip()

                # Subject identification
                is_subject = speaker_clean.lower() in [
                    'subj_533', 'user', 'subject_533', 'subject533'
                ]

                current_msg = {
                    'timestamp': None,  # ChatGPT exports often lack timestamps
                    'speaker': speaker_clean,
                    'text': text.strip(),
                    'msg_index': msg_index,
                    'is_subject': is_subject
                }
                msg_index += 1
            else:
                # Continuation line
                if current_msg and line.strip():
                    current_msg['text'] += ' ' + line.strip()

    if current_msg:
        messages.append(current_msg)

    subject_msgs = [m for m in messages if m['is_subject']]
    print(f"[ChatGPT] Parsed {len(messages)} total messages, "
          f"{len(subject_msgs)} from subject")
    return subject_msgs


# =============================================================================
# NRC EMOTION EXTRACTION (LETA METHOD)
# =============================================================================

EMOTION_CATEGORIES = [
    'anger', 'anticipation', 'disgust', 'fear',
    'joy', 'sadness', 'surprise', 'trust',
    'negative', 'positive'
]

def extract_emotions(text, lexicon):
    """
    Apply NRC lexicon to text. Returns dict of emotion scores.
    Score = count of emotion words / total words (proportion).
    Also returns per-1000-token rate for PersonaDrift compatibility.
    """
    # Tokenize: lowercase, word-boundary split
    tokens = re.findall(r'\b[a-zA-Z\']+\b', text.lower())
    total_tokens = len(tokens)

    if total_tokens == 0:
        result = {f'score_{e}': 0.0 for e in EMOTION_CATEGORIES}
        result.update({f'count_{e}': 0 for e in EMOTION_CATEGORIES})
        result.update({f'per1k_{e}': 0.0 for e in EMOTION_CATEGORIES})
        result['total_tokens'] = 0
        result['emotion_tokens'] = 0
        return result

    # Count emotions
    counts = {e: 0 for e in EMOTION_CATEGORIES}
    emotion_token_count = 0

    for token in tokens:
        if token in lexicon:
            emotion_token_count += 1
            for emotion in lexicon[token]:
                if emotion in counts:
                    counts[emotion] += 1

    # Compute scores
    result = {}
    for e in EMOTION_CATEGORIES:
        result[f'score_{e}'] = counts[e] / total_tokens  # LETA format (proportion)
        result[f'count_{e}'] = counts[e]                   # raw count
        result[f'per1k_{e}'] = (counts[e] / total_tokens) * 1000  # PersonaDrift format

    result['total_tokens'] = total_tokens
    result['emotion_tokens'] = emotion_token_count

    # Primary emotion (excluding positive/negative aggregates)
    core_emotions = ['anger', 'anticipation', 'disgust', 'fear',
                     'joy', 'sadness', 'surprise', 'trust']
    core_scores = {e: result[f'score_{e}'] for e in core_emotions}
    sorted_emotions = sorted(core_scores.items(), key=lambda x: x[1], reverse=True)

    result['primary_emotion'] = sorted_emotions[0][0]
    result['primary_intensity'] = sorted_emotions[0][1]
    result['secondary_emotion'] = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
    result['tertiary_emotion'] = sorted_emotions[2][0] if len(sorted_emotions) > 2 else None

    return result


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_environment(filepath, env_name, parse_format, lexicon, subject_ids):
    """Process a single environment file through LETA extraction."""

    print(f"\n{'='*60}")
    print(f"Processing: {env_name} ({parse_format})")
    print(f"File: {filepath}")
    print(f"{'='*60}")

    # Parse based on format
    if parse_format == 'whatsapp':
        messages = parse_whatsapp_export(filepath, subject_ids)
    elif parse_format == 'chatgpt':
        messages = parse_chatgpt_export(filepath, subject_tag=subject_ids[0])
    else:
        raise ValueError(f"Unknown format: {parse_format}")

    # Filter empty messages
    messages = [m for m in messages if m['text'].strip() and len(m['text'].split()) >= 3]
    print(f"[Filter] {len(messages)} messages with 3+ words")

    # Extract emotions for each message
    rows = []
    for i, msg in enumerate(messages):
        emotions = extract_emotions(msg['text'], lexicon)
        row = {
            'environment': env_name,
            'msg_index': i,
            'original_index': msg['msg_index'],
            'timestamp': msg.get('timestamp'),
            'word_count': emotions['total_tokens'],
            'text_preview': msg['text'][:100] + '...' if len(msg['text']) > 100 else msg['text'],
        }
        row.update(emotions)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Summary stats
    print(f"\n[Summary] {env_name}:")
    print(f"  Messages: {len(df)}")
    print(f"  Total tokens: {df['total_tokens'].sum():,}")
    print(f"  Mean message length: {df['total_tokens'].mean():.1f} words")
    print(f"  Top primary emotions: {df['primary_emotion'].value_counts().head(3).to_dict()}")

    return df


def main():
    parser = argparse.ArgumentParser(description='LETA x PersonaDrift: Extract Emotions')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    nrc_path = config['nrc_lexicon_path']
    output_path = config.get('output_csv', 'persona_drift_leta_emotions.csv')
    environments = config['environments']

    # Load NRC lexicon
    lexicon = load_nrc_lexicon(nrc_path)

    # Process each environment
    all_dfs = []
    for env in environments:
        df = process_environment(
            filepath=env['filepath'],
            env_name=env['name'],
            parse_format=env['format'],
            lexicon=lexicon,
            subject_ids=env.get('subject_identifiers', ['Subj_533'])
        )
        all_dfs.append(df)

    # Combine and save
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"SAVED: {output_path}")
    print(f"Total rows: {len(combined)}")
    print(f"Environments: {combined['environment'].value_counts().to_dict()}")
    print(f"{'='*60}")

    # Also save a summary table
    summary = combined.groupby('environment').agg({
        'msg_index': 'count',
        'total_tokens': ['sum', 'mean'],
        'score_anger': 'mean',
        'score_joy': 'mean',
        'score_sadness': 'mean',
        'score_fear': 'mean',
        'score_trust': 'mean',
        'score_anticipation': 'mean',
    }).round(4)

    summary_path = output_path.replace('.csv', '_summary.csv')
    summary.to_csv(summary_path)
    print(f"SAVED: {summary_path}")


if __name__ == '__main__':
    main()