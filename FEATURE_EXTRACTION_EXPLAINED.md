# Parkinson's Disease Voice Analysis: Feature Extraction Pipeline Explained

## 📋 Project Overview

This project extracts acoustic features from voice recordings to help analyze and distinguish between Parkinson's Disease (PD) and Healthy Control (HC) subjects. The extracted features are used for downstream statistical analysis and machine learning.

---

## 📁 Dataset Structure & Labeling

- **Audio files** are stored in subfolders under:
  - `Processed_data_sample_raw_voice/raw_wav/0/` and `Processed_data_sample_raw_voice/raw_wav/1/`
- **Important:**
  - The folder names `0/` and `1/` are **arbitrary** and do **not** represent PD or HC labels.
  - The **true label** for each audio file is found in `all_audios_mapped_id_for_label/final_selected.csv` in the `cohort` column.
- **How it works:**
  - Each row in `final_selected.csv` contains metadata for one audio file, including:
    - `audio_audio.m4a`: The audio file's unique ID
    - `cohort`: The label (`PD` or `Control`)

---

## 🛠️ Feature Extraction Pipeline: Step-by-Step

1. **Read the metadata CSV** (`final_selected.csv`) into a DataFrame.
2. **For each row** (audio sample):
   - Extract the `audio_id` from `audio_audio.m4a`.
   - Use this ID to find the corresponding `.wav` file (searches both `0/` and `1/` folders).
   - Extract features from the audio file.
   - The label for this file is always `row['cohort']` from the CSV.
3. **Save the extracted features** (and metadata) to a new CSV for each feature type.
4. **Merge**: The output CSVs retain the original metadata, including the label.

---

## 🎯 Features Extracted (Per Audio File)

### 1. Jitter (Frequency Perturbation)

- **What:** Measures cycle-to-cycle variation in pitch (voice stability).
- **How computed:** Several metrics, each producing one value for the whole file:
  - `jitter_local`, `jitter_rap`, `jitter_ppq5`, `jitter_ddp`, `jitter_manual`
- **Output:** 5 values per audio file (one per metric).
- **Example:**
  | audio_id | jitter_local | jitter_rap | jitter_ppq5 | jitter_ddp | jitter_manual |
  |----------|--------------|------------|-------------|------------|--------------|
  | 5394000 | 0.0041 | 0.0023 | 0.0030 | 0.0018 | 0.0052 |

### 2. Shimmer (Amplitude Perturbation)

- **What:** Measures cycle-to-cycle variation in amplitude (voice clarity).
- **How computed:** Several metrics, each producing one value for the whole file:
  - `shimmer_local`, `shimmer_apq3`, `shimmer_apq5`, `shimmer_apq11`, `shimmer_manual`
- **Output:** 5 values per audio file.
- **Example:**
  | audio_id | shimmer_local | shimmer_apq3 | shimmer_apq5 | shimmer_apq11 | shimmer_manual |
  |----------|--------------|--------------|--------------|---------------|---------------|
  | 5394000 | 0.021 | 0.012 | 0.015 | 0.018 | 0.022 |

### 3. Fundamental Frequency (F0)

- **What:** The pitch of the voice.
- **How computed:** Summary statistics over the whole file:
  - `f0_mean`, `f0_min`, `f0_max`, `f0_range`, `f0_std`
- **Output:** 5 values per audio file.
- **Example:**
  | audio_id | f0_mean | f0_min | f0_max | f0_range | f0_std |
  |----------|--------|--------|--------|----------|--------|
  | 5394000 | 157.5 | 120.0 | 180.0 | 60.0 | 13.1 |

### 4. Harmonics-to-Noise Ratio (HNR)

- **What:** Ratio of harmonic to noise components (voice clarity).
- **How computed:** Three methods, each producing one value for the whole file:
  - `hnr_autocorr`, `hnr_cepstral`, `hnr_manual`
- **Output:** Up to 3 values per audio file.
- **Example:**
  | audio_id | hnr_autocorr | hnr_cepstral | hnr_manual |
  |----------|--------------|--------------|------------|
  | 5394000 | 18.2 | 17.9 | 19.1 |

### 5. Zero-Crossing Rate (ZCR)

- **What:** Rate at which the signal changes sign (noisiness).
- **How computed:**
  - `zcr_overall`: For the whole file
  - File is split into 10 segments; summary stats are computed:
    - `zcr_mean`, `zcr_std`, `zcr_min`, `zcr_max`
- **Output:** 5 values per audio file.
- **Example:**
  | audio_id | zcr_overall | zcr_mean | zcr_std | zcr_min | zcr_max |
  |----------|-------------|----------|---------|---------|---------|
  | 5394000 | 0.042 | 0.045 | 0.003 | 0.041 | 0.049 |

### 6. Voice Breaks / Unvoiced Segments

- **What:** Measures of voiced/unvoiced segments, voice breaks, durations, etc.
- **How computed:** Summary statistics for the whole file:
  - `voice_breaks_count`, `voiced_percentage`, `unvoiced_percentage`, `avg_voiced_duration`, `avg_unvoiced_duration`, `voiced_segments_count`, `unvoiced_segments_count`
- **Output:** 7 values per audio file.
- **Example:**
  | audio_id | voice_breaks_count | voiced_percentage | unvoiced_percentage | avg_voiced_duration | avg_unvoiced_duration | voiced_segments_count | unvoiced_segments_count |
  |----------|--------------------|-------------------|---------------------|---------------------|-----------------------|----------------------|------------------------|
  | 5394000 | 3 | 85.2 | 14.8 | 1.2 | 0.3 | 4 | 3 |

---

## 🗂️ Output CSVs & Label Preservation

- Each feature extraction script saves a CSV with extracted features and metadata.
- The output CSVs are merged with the original `final_selected.csv`, so the `cohort` (label) column is always present for downstream analysis.
- **You never need to use the folder name for labeling.**

---

## 📊 Quick Reference Table

| Feature      | # Values per File | Per-Frame/Segment? | Output Columns (examples)         |
| ------------ | :---------------: | :----------------: | --------------------------------- |
| Jitter       |         5         |    No (summary)    | jitter_local, jitter_rap, ...     |
| Shimmer      |         5         |    No (summary)    | shimmer_local, shimmer_apq3, ...  |
| F0           |         5         |    No (summary)    | f0_mean, f0_min, ...              |
| HNR          |      up to 3      |    No (summary)    | hnr_autocorr, hnr_cepstral, ...   |
| ZCR          |         5         |    No (summary)    | zcr_overall, zcr_mean, ...        |
| Voice Breaks |         7         |    No (summary)    | voice*breaks_count, voiced*%, ... |

---

## 🧬 Clinical Relevance: Feature Trends in Parkinson's Disease

| Feature       | PD Tends To Cause...        | Interpretation for PD Detection      |
| ------------- | --------------------------- | ------------------------------------ |
| Jitter        | Higher values               | ↑ Jitter → More likely PD            |
| Shimmer       | Higher values               | ↑ Shimmer → More likely PD           |
| F0 mean/range | Lower values                | ↓ F0 mean/range → More likely PD     |
| HNR           | Lower values                | ↓ HNR → More likely PD               |
| ZCR           | Higher values (sometimes)   | ↑ ZCR → May indicate PD              |
| Voice Breaks  | More frequent/longer breaks | ↑ Breaks/unvoiced % → More likely PD |

**Notes:**

- Higher jitter and shimmer indicate more unstable, rough, or breathy voice, common in PD.
- Lower F0 mean/range reflects monotone or reduced pitch variation, typical in PD.
- Lower HNR means a noisier, less clear voice, often seen in PD.
- More voice breaks and higher unvoiced percentage reflect difficulty sustaining phonation.
- ZCR can be higher in PD due to increased noise, but is less specific than other features.

---

## 🧑‍💻 Code Snippets: How Features and Labels Are Handled

### 1. **Reading the Metadata and Label**

```python
import pandas as pd
CSV_PATH = "all_audios_mapped_id_for_label/final_selected.csv"
df = pd.read_csv(CSV_PATH, sep='\t')
if len(df.columns) == 1:
    df = pd.read_csv(CSV_PATH, sep=',')

for idx, row in df.iterrows():
    audio_id = str(row['audio_audio.m4a'])
    label = row['cohort']  # 'PD' or 'Control'
    # ...
```

- **Explanation:** The label for each audio file is always taken from the `cohort` column in the CSV, never from the folder name.

### 2. **Finding the Audio File**

```python
import os
def find_audio_path(base_dir, audio_id):
    for subdir in ['0', '1']:
        search_dir = os.path.join(base_dir, subdir)
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.wav') and audio_id in file:
                    return os.path.join(root, file)
    return None
```

- **Explanation:** The code searches both `0/` and `1/` folders for the audio file, but does not use the folder name for labeling.

### 3. **Extracting Features (Example: Jitter)**

```python
from utils import extract_jitter
jitter_values = extract_jitter(audio_path)
# jitter_values is a dict with keys: 'jitter_local', 'jitter_rap', ...
```

- **Explanation:** Each feature extraction function returns a dictionary of summary values for the whole file.

### 4. **Saving Results with Labels**

```python
import pandas as pd
import os
results = []
for idx, row in df.iterrows():
    audio_id = str(row['audio_audio.m4a'])
    label = row['cohort']
    audio_path = find_audio_path("Processed_data_sample_raw_voice/raw_wav", audio_id)
    if audio_path:
        try:
            jitter_values = extract_jitter(audio_path)
            results.append({
                'audio_id': audio_id,
                'jitter_local': jitter_values['jitter_local'],
                'jitter_rap': jitter_values['jitter_rap'],
                'jitter_ppq5': jitter_values['jitter_ppq5'],
                'jitter_ddp': jitter_values['jitter_ddp'],
                'jitter_manual': jitter_values['jitter_manual'],
                'label': label
            })
        except Exception as e:
            print(f"Error extracting features for {audio_id}: {e}")
    else:
        print(f"Audio file not found for {audio_id}")

result_df = pd.DataFrame(results)
result_df.to_csv('features/jitter_features.csv', index=False)
```

- **Explanation:** The output CSV includes both the extracted features and the true label for each file.

---

