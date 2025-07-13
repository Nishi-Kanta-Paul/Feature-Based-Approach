# Parkinson's Disease Voice Analysis - Acoustic Feature Extraction Pipeline

## 📋 Project Overview

This project implements a comprehensive acoustic feature extraction pipeline for analyzing voice samples in Parkinson's Disease research. The system extracts multiple acoustic features from audio files to help identify voice characteristics that may indicate Parkinson's Disease.

## 🎯 Features Extracted

### 1. **Jitter (Frequency Perturbation)**
- **Definition**: Variation in fundamental frequency (F0) from cycle to cycle
- **Use**: Measures voice stability and hoarseness
- **Methods**: Local, RAP, PPQ5, DDP, and manual calculation
- **Output**: Multiple jitter metrics for each audio file

### 2. **Shimmer (Amplitude Perturbation)**
- **Definition**: Variation in amplitude from cycle to cycle
- **Use**: Measures voice clarity and breathiness
- **Methods**: Local, APQ3, APQ5, APQ11, and manual calculation
- **Output**: Multiple shimmer metrics for each audio file

### 3. **Fundamental Frequency (F0)**
- **Definition**: The pitch of the voice, measured in Hz
- **Use**: Helps in speaker identification, emotion detection, gender recognition
- **Statistics**: Mean, min, max, range, standard deviation
- **Output**: Comprehensive F0 statistics for each audio file

### 4. **Harmonics-to-Noise Ratio (HNR)**
- **Definition**: Ratio of periodic (harmonic) to aperiodic (noise) components
- **Use**: Measures voice clarity. Lower HNR = breathy or hoarse voice
- **Methods**: Autocorrelation, cepstral, and manual spectral analysis
- **Output**: HNR values in decibels (dB)

### 5. **Zero-Crossing Rate (ZCR)**
- **Definition**: Rate at which the signal changes sign
- **Use**: Detects noisy or unvoiced segments; useful for speech/music separation
- **Statistics**: Overall, mean, std, min, max across segments
- **Output**: ZCR metrics for signal analysis

### 6. **Voice Breaks / Unvoiced Segments**
- **Definition**: Duration or % of time the voice signal is not voiced (silent or unvoiced)
- **Use**: Useful for detecting speech pathologies or affective states
- **Metrics**: Voice breaks count, voiced/unvoiced percentages, segment durations
- **Output**: Comprehensive voice segmentation analysis

## 📁 Project Structure

```
Feature-Based-Approach/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── src/                                         # Source code directory
│   ├── __init__.py
│   ├── utils.py                                 # Core utility functions
│   ├── extract_jitter.py                        # Jitter extraction script
│   ├── extract_shimmer.py                       # Shimmer extraction script
│   ├── extract_f0.py                           # F0 extraction script
│   ├── extract_hnr.py                          # HNR extraction script
│   ├── extract_zcr.py                          # ZCR extraction script
│   └── extract_voice_breaks.py                 # Voice breaks extraction script
├── features/                                    # Output directory for extracted features
│   ├── jitter_features.csv                     # Jitter extraction results
│   ├── shimmer_features.csv                    # Shimmer extraction results
│   ├── f0_features.csv                         # F0 extraction results
│   ├── hnr_features.csv                        # HNR extraction results
│   ├── zcr_features.csv                        # ZCR extraction results
│   ├── voice_breaks_features.csv               # Voice breaks extraction results
│   ├── jitter_extraction_errors.log            # Jitter extraction error log
│   ├── shimmer_extraction_errors.log           # Shimmer extraction error log
│   ├── f0_extraction_errors.log                # F0 extraction error log
│   ├── hnr_extraction_errors.log               # HNR extraction error log
│   ├── zcr_extraction_errors.log               # ZCR extraction error log
│   └── voice_breaks_extraction_errors.log      # Voice breaks extraction error log
├── all_audios_mapped_id_for_label/             # Input data directory
│   └── final_selected.csv                      # CSV file with audio IDs and metadata
└── Processed_data_sample_raw_voice/            # Audio files directory
    └── raw_wav/                                # Raw WAV audio files
        ├── 0/                                  # Arbitrary folder name (does NOT indicate label)
        │   ├── 5394000/
        │   │   └── audio_audio.m4a-[hash].wav
        │   └── 5395000/
        │       └── audio_audio.m4a-[hash].wav
        └── 1/                                  # Arbitrary folder name (does NOT indicate label)
            ├── 5389001/
            │   └── audio_audio.m4a-[hash].wav
            └── 5392001/
                └── audio_audio.m4a-[hash].wav
```

> **Note:**
> The folder names `0/` and `1/` under `Processed_data_sample_raw_voice/raw_wav/` are arbitrary and do **not** represent PD or HC labels. **Do not use these folder names for classification.**
> 
> The true label (PD or HC) for each audio file is provided in the `final_selected.csv` file, specifically in the `cohort` column. Always use the CSV for label information.

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Windows/Linux/macOS

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd Feature-Based-Approach
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `pandas`: Data manipulation and CSV handling
- `parselmouth`: Python interface to Praat for acoustic analysis
- `numpy`: Numerical computations
- `os`: File system operations

## 📊 Data Format

### Input CSV Format (`final_selected.csv`)
The system expects a CSV file with the following columns:
- `ROW_ID`: Unique row identifier
- `ROW_VERSION`: Version number
- `recordId`: Record identifier
- `healthCode`: Health code
- `createdOn`: Creation timestamp
- `appVersion`: App version
- `phoneInfo`: Phone information
- `audio_audio.m4a`: Audio ID (numeric)
- `audio_countdown.m4a`: Countdown audio ID
- `medTimepoint`: Medication timepoint
- `cohort`: Patient cohort (PD/Control)

### Audio File Structure
Audio files should be organized as:
```
Processed_data_sample_raw_voice/raw_wav/
├── 0/                                          # Class 0 (Control)
│   ├── 5394000/
│   │   └── audio_audio.m4a-[hash].wav
│   └── 5395000/
│       └── audio_audio.m4a-[hash].wav
└── 1/                                          # Class 1 (PD)
    ├── 5389001/
    │   └── audio_audio.m4a-[hash].wav
    └── 5392001/
        └── audio_audio.m4a-[hash].wav
```

## 🔧 Usage

### Running Individual Feature Extraction Scripts

#### 1. Jitter Extraction
```bash
python src/extract_jitter.py
```
**Output**: `features/jitter_features.csv`
**Features**: jitter_local, jitter_rap, jitter_ppq5, jitter_ddp, jitter_manual

#### 2. Shimmer Extraction
```bash
python src/extract_shimmer.py
```
**Output**: `features/shimmer_features.csv`
**Features**: shimmer_local, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_manual

#### 3. Fundamental Frequency (F0) Extraction
```bash
python src/extract_f0.py
```
**Output**: `features/f0_features.csv`
**Features**: f0_mean, f0_min, f0_max, f0_range, f0_std

#### 4. Harmonics-to-Noise Ratio (HNR) Extraction
```bash
python src/extract_hnr.py
```
**Output**: `features/hnr_features.csv`
**Features**: hnr_autocorr, hnr_cepstral, hnr_manual

#### 5. Zero-Crossing Rate (ZCR) Extraction
```bash
python src/extract_zcr.py
```
**Output**: `features/zcr_features.csv`
**Features**: zcr_overall, zcr_mean, zcr_std, zcr_min, zcr_max

#### 6. Voice Breaks Extraction
```bash
python src/extract_voice_breaks.py
```
**Output**: `features/voice_breaks_features.csv`
**Features**: voice_breaks_count, voiced_percentage, unvoiced_percentage, avg_voiced_duration, avg_unvoiced_duration, voiced_segments_count, unvoiced_segments_count

### Running All Extractions
```bash
# Run all feature extractions sequentially
python src/extract_jitter.py
python src/extract_shimmer.py
python src/extract_f0.py
python src/extract_hnr.py
python src/extract_zcr.py
python src/extract_voice_breaks.py
```

## 📈 Output Analysis

### Success Metrics
Each script provides detailed statistics:
- **Total files processed**: Number of audio IDs in CSV
- **Files found**: Number of audio files actually located
- **Successful extractions**: Number of successful feature extractions
- **Failed extractions**: Number of failed extractions
- **Success rate**: Percentage of successful extractions

### Sample Output
```
📊 F0 Extraction Summary:
   Total files processed: 55939
   Files found: 4
   Successful extractions: 4
   Failed extractions: 55935
   Success rate: 0.0%

📈 Sample F0 Statistics:
   Average F0 Mean: 157.49 Hz
   Average F0 Range: 100.01 Hz
   Average F0 Std: 13.14 Hz

🎯 Top 5 F0 Values:
   6042405: Mean=240.26Hz, Range=26.05Hz
   5410698: Mean=144.83Hz, Range=260.00Hz
   5522310: Mean=125.25Hz, Range=68.56Hz
   5843512: Mean=119.61Hz, Range=45.41Hz
```

## 🔍 Error Handling

### Error Logs
Each extraction script creates detailed error logs in the `features/` directory:
- `*_extraction_errors.log`: Detailed error messages for each failed extraction
- Common errors include:
  - Audio file not found
  - Insufficient voiced segments
  - Praat command failures
  - Invalid audio format

### Robust Extraction Methods
The system implements multiple fallback methods:
- **Primary methods**: Standard Praat algorithms
- **Fallback methods**: Manual calculations using signal processing
- **Error recovery**: Graceful handling of extraction failures

## 🛠️ Technical Details

### Core Functions (`src/utils.py`)

#### `find_audio_path(base_dir, audio_id)`
- Recursively searches for audio files containing the audio_id
- Searches in both '0' and '1' subdirectories
- Returns full path to the audio file or None if not found

#### `extract_jitter(audio_path)`
- Extracts jitter using multiple Praat methods
- Implements manual calculation as fallback
- Returns dictionary with all jitter metrics

#### `extract_shimmer(audio_path)`
- Extracts shimmer using multiple Praat methods
- Implements manual amplitude variation calculation
- Returns dictionary with all shimmer metrics

#### `extract_fundamental_frequency(audio_path)`
- Extracts F0 using Praat pitch analysis
- Calculates comprehensive statistics
- Returns mean, min, max, range, and standard deviation

#### `extract_hnr(audio_path)`
- Extracts HNR using autocorrelation and cepstral methods
- Implements manual spectral analysis
- Returns HNR values in decibels

#### `extract_zero_crossing_rate(audio_path)`
- Calculates ZCR for entire signal and segments
- Provides statistical analysis across segments
- Returns overall and segment-wise ZCR metrics

#### `extract_voice_breaks(audio_path)`
- Analyzes voiced vs unvoiced segments
- Calculates voice breaks and segment statistics
- Returns comprehensive voice segmentation metrics

### Algorithm Details

#### Jitter Calculation
1. **Praat Methods**: Uses PointProcess objects for standard jitter metrics
2. **Manual Method**: Calculates coefficient of variation of pitch values
3. **Fallback**: Handles cases with insufficient voiced segments

#### Shimmer Calculation
1. **Praat Methods**: Uses PointProcess objects for standard shimmer metrics
2. **Manual Method**: Calculates coefficient of variation of amplitude values
3. **Fallback**: Analyzes amplitude at voiced pitch points

#### F0 Analysis
1. **Pitch Extraction**: Uses Praat's pitch analysis algorithm
2. **Statistics**: Calculates comprehensive F0 statistics
3. **Validation**: Ensures sufficient voiced segments for analysis

#### HNR Analysis
1. **Autocorrelation**: Standard Praat HNR method
2. **Cepstral**: Alternative HNR calculation method
3. **Manual**: Spectral analysis of harmonic vs noise components

#### ZCR Analysis
1. **Signal Processing**: Calculates zero-crossing rate across entire signal
2. **Segmentation**: Divides signal into 10 segments for analysis
3. **Statistics**: Provides comprehensive ZCR statistics

#### Voice Breaks Analysis
1. **Segmentation**: Identifies voiced vs unvoiced segments
2. **Transitions**: Counts voice breaks (voiced to unvoiced transitions)
3. **Statistics**: Calculates segment durations and percentages

## 📊 Data Validation

### Input Validation
- CSV format detection (tab vs comma separated)
- Audio ID extraction and validation
- File existence verification

### Output Validation
- Feature value range checking
- Statistical outlier detection
- Missing value handling

## 🔧 Customization

### Modifying Extraction Parameters
Edit the constants in each extraction script:
```python
CSV_PATH = "all_audios_mapped_id_for_label/final_selected.csv"
AUDIO_BASE = "Processed_data_sample_raw_voice/raw_wav"
OUTPUT_PATH = "features/[feature]_features.csv"
LOG_PATH = "features/[feature]_extraction_errors.log"
```

### Adding New Features
1. Create new extraction function in `src/utils.py`
2. Create new extraction script following the existing pattern
3. Update this README with new feature documentation

## 🐛 Troubleshooting

### Common Issues

#### 1. Audio Files Not Found
- **Cause**: Audio IDs in CSV don't match actual file names
- **Solution**: Check file naming convention and update search logic

#### 2. Praat Command Failures
- **Cause**: Insufficient voiced segments or poor audio quality
- **Solution**: Check audio quality and adjust minimum segment requirements

#### 3. Memory Issues
- **Cause**: Large audio files or too many files processed simultaneously
- **Solution**: Process files in batches or increase system memory

#### 4. Import Errors
- **Cause**: Missing dependencies
- **Solution**: Run `pip install -r requirements.txt`

### Performance Optimization
- Process files in batches for large datasets
- Use multiprocessing for parallel extraction
- Optimize audio file loading and processing

## 📚 References

### Academic Papers
- Jitter and Shimmer measurements in voice analysis
- Fundamental frequency analysis in Parkinson's Disease
- Harmonics-to-Noise Ratio in voice pathology
- Zero-Crossing Rate applications in speech analysis

### Technical Documentation
- Praat Manual: https://www.fon.hum.uva.nl/praat/
- Parselmouth Documentation: https://parselmouth.readthedocs.io/
- NumPy Documentation: https://numpy.org/doc/
- Pandas Documentation: https://pandas.pydata.org/docs/

## 🤝 Contributing

### Development Guidelines
1. Follow existing code structure and naming conventions
2. Add comprehensive error handling
3. Include detailed logging
4. Update documentation for new features
5. Test with sample data before deployment

### Code Style
- Use descriptive variable names
- Include docstrings for all functions
- Follow PEP 8 style guidelines
- Add type hints where appropriate

## 📄 License

This project is developed for research purposes in Parkinson's Disease voice analysis.

## 📞 Support

For questions or issues:
1. Check the error logs in the `features/` directory
2. Review the troubleshooting section
3. Verify input data format and file structure
4. Contact the development team for technical support

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Compatibility**: Python 3.7+, Windows/Linux/macOS
