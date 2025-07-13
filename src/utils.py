import os
import parselmouth
import numpy as np


def find_audio_path(base_dir, audio_id):
    """Recursively search all subfolders under '0' and '1' for a .wav file containing audio_id in its name."""
    for subdir in ['0', '1']:
        search_dir = os.path.join(base_dir, subdir)
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.wav') and audio_id in file:
                    print(
                        f"Found file for {audio_id}: {os.path.join(root, file)}")
                    return os.path.join(root, file)
    print(
        f"No .wav file found for audio_id {audio_id} in any subfolder of '0' or '1'")
    return None


def find_all_audio_paths(base_dir, audio_ids):
    """Find audio paths for all audio IDs."""
    audio_paths = {}
    for audio_id in audio_ids:
        path = find_audio_path(base_dir, audio_id)
        if path:
            audio_paths[audio_id] = path
    return audio_paths


def extract_jitter(audio_path):
    """Extract jitter (frequency perturbation) from audio file."""
    try:
        # Load audio file
        sound = parselmouth.Sound(audio_path)

        # Extract pitch
        pitch = sound.to_pitch()

        # Extract PointProcess from pitch
        point_process = pitch.to_point_process()

        # Try different jitter extraction methods
        jitter_values = {}

        # Method 1: Jitter (local)
        try:
            jitter_local = point_process.get_jitter_local()
            jitter_values['jitter_local'] = jitter_local
        except:
            jitter_values['jitter_local'] = None

        # Method 2: Jitter (rap)
        try:
            jitter_rap = point_process.get_jitter_rap()
            jitter_values['jitter_rap'] = jitter_rap
        except:
            jitter_values['jitter_rap'] = None

        # Method 3: Jitter (ppq5)
        try:
            jitter_ppq5 = point_process.get_jitter_ppq5()
            jitter_values['jitter_ppq5'] = jitter_ppq5
        except:
            jitter_values['jitter_ppq5'] = None

        # Method 4: Jitter (ddp)
        try:
            jitter_ddp = point_process.get_jitter_ddp()
            jitter_values['jitter_ddp'] = jitter_ddp
        except:
            jitter_values['jitter_ddp'] = None

        # Method 5: Manual calculation using pitch variation
        try:
            pitch_values = pitch.selected_array['frequency']
            voiced_pitch = pitch_values[pitch_values > 0]
            if len(voiced_pitch) > 5:
                # Calculate jitter as coefficient of variation of pitch
                jitter_manual = np.std(voiced_pitch) / np.mean(voiced_pitch)
                jitter_values['jitter_manual'] = jitter_manual
            else:
                jitter_values['jitter_manual'] = None
        except:
            jitter_values['jitter_manual'] = None

        return jitter_values

    except Exception as e:
        return {key: None for key in ['jitter_local', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'jitter_manual']}


def extract_shimmer(audio_path):
    """Extract shimmer (amplitude perturbation) from audio file."""
    try:
        # Load audio file
        sound = parselmouth.Sound(audio_path)

        # Extract pitch
        pitch = sound.to_pitch()

        # Extract PointProcess from pitch
        point_process = pitch.to_point_process()

        # Try different shimmer extraction methods
        shimmer_values = {}

        # Method 1: Shimmer (local)
        try:
            shimmer_local = point_process.get_shimmer_local()
            shimmer_values['shimmer_local'] = shimmer_local
        except:
            shimmer_values['shimmer_local'] = None

        # Method 2: Shimmer (apq3)
        try:
            shimmer_apq3 = point_process.get_shimmer_apq3()
            shimmer_values['shimmer_apq3'] = shimmer_apq3
        except:
            shimmer_values['shimmer_apq3'] = None

        # Method 3: Shimmer (apq5)
        try:
            shimmer_apq5 = point_process.get_shimmer_apq5()
            shimmer_values['shimmer_apq5'] = shimmer_apq5
        except:
            shimmer_values['shimmer_apq5'] = None

        # Method 4: Shimmer (apq11)
        try:
            shimmer_apq11 = point_process.get_shimmer_apq11()
            shimmer_values['shimmer_apq11'] = shimmer_apq11
        except:
            shimmer_values['shimmer_apq11'] = None

        # Method 5: Manual calculation using amplitude variation
        try:
            # Get amplitude values at pitch points
            pitch_values = pitch.selected_array['frequency']
            voiced_indices = np.where(pitch_values > 0)[0]

            if len(voiced_indices) > 5:
                # Get amplitude at voiced points
                amplitude_values = []
                for idx in voiced_indices:
                    time = pitch.x1 + idx * pitch.dx
                    if time < sound.duration:
                        amplitude = sound.get_value_at_time(time)
                        if not np.isnan(amplitude):
                            amplitude_values.append(abs(amplitude))

                if len(amplitude_values) > 5:
                    # Calculate shimmer as coefficient of variation of amplitude
                    shimmer_manual = np.std(
                        amplitude_values) / np.mean(amplitude_values)
                    shimmer_values['shimmer_manual'] = shimmer_manual
                else:
                    shimmer_values['shimmer_manual'] = None
            else:
                shimmer_values['shimmer_manual'] = None
        except:
            shimmer_values['shimmer_manual'] = None

        return shimmer_values

    except Exception as e:
        return {key: None for key in ['shimmer_local', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_manual']}


def extract_fundamental_frequency(audio_path):
    """Extract Fundamental Frequency (F0) statistics from audio file."""
    try:
        # Load audio file
        sound = parselmouth.Sound(audio_path)

        # Extract pitch
        pitch = sound.to_pitch()

        # Get pitch values
        pitch_values = pitch.selected_array['frequency']
        voiced_pitch = pitch_values[pitch_values > 0]

        if len(voiced_pitch) == 0:
            return {
                'f0_mean': None,
                'f0_min': None,
                'f0_max': None,
                'f0_range': None,
                'f0_std': None
            }

        # Calculate F0 statistics
        f0_mean = np.mean(voiced_pitch)
        f0_min = np.min(voiced_pitch)
        f0_max = np.max(voiced_pitch)
        f0_range = f0_max - f0_min
        f0_std = np.std(voiced_pitch)

        return {
            'f0_mean': f0_mean,
            'f0_min': f0_min,
            'f0_max': f0_max,
            'f0_range': f0_range,
            'f0_std': f0_std
        }

    except Exception as e:
        return {
            'f0_mean': None,
            'f0_min': None,
            'f0_max': None,
            'f0_range': None,
            'f0_std': None
        }


def extract_hnr(audio_path):
    """Extract Harmonics-to-Noise Ratio (HNR) from audio file."""
    try:
        # Load audio file
        sound = parselmouth.Sound(audio_path)

        # Extract pitch
        pitch = sound.to_pitch()

        # Extract PointProcess from pitch
        point_process = pitch.to_point_process()

        # Try different HNR extraction methods
        hnr_values = {}

        # Method 1: HNR (autocorrelation)
        try:
            hnr_autocorr = point_process.get_harmonicity_autocorrelation()
            hnr_values['hnr_autocorr'] = hnr_autocorr
        except:
            hnr_values['hnr_autocorr'] = None

        # Method 2: HNR (cepstral)
        try:
            hnr_cepstral = point_process.get_harmonicity_cepstral()
            hnr_values['hnr_cepstral'] = hnr_cepstral
        except:
            hnr_values['hnr_cepstral'] = None

        # Method 3: Manual calculation using spectral analysis
        try:
            # Get voiced segments
            pitch_values = pitch.selected_array['frequency']
            voiced_indices = np.where(pitch_values > 0)[0]

            if len(voiced_indices) > 10:
                # Calculate HNR for voiced segments
                hnr_manual_values = []
                for i in range(0, len(voiced_indices), max(1, len(voiced_indices)//10)):
                    idx = voiced_indices[i]
                    time = pitch.x1 + idx * pitch.dx
                    if time < sound.duration:
                        # Extract spectrum at this time
                        spectrum = sound.to_spectrum_at_time(time)
                        if spectrum:
                            # Calculate harmonic and noise components
                            frequencies = spectrum.xs()
                            powers = spectrum.ys()

                            # Find fundamental frequency
                            f0 = pitch_values[idx]
                            if f0 > 0:
                                # Find harmonics
                                harmonic_power = 0
                                noise_power = 0

                                for j, freq in enumerate(frequencies):
                                    # Check if frequency is near a harmonic of F0
                                    harmonic_number = round(freq / f0)
                                    if harmonic_number > 0 and harmonic_number <= 5:  # First 5 harmonics
                                        # Within 10% tolerance
                                        if abs(freq - harmonic_number * f0) < f0 * 0.1:
                                            harmonic_power += powers[j]
                                        else:
                                            noise_power += powers[j]
                                    else:
                                        noise_power += powers[j]

                                if noise_power > 0:
                                    hnr = 10 * \
                                        np.log10(harmonic_power / noise_power)
                                    hnr_manual_values.append(hnr)

                if len(hnr_manual_values) > 0:
                    hnr_values['hnr_manual'] = np.mean(hnr_manual_values)
                else:
                    hnr_values['hnr_manual'] = None
            else:
                hnr_values['hnr_manual'] = None
        except:
            hnr_values['hnr_manual'] = None

        return hnr_values

    except Exception as e:
        return {key: None for key in ['hnr_autocorr', 'hnr_cepstral', 'hnr_manual']}


def extract_zero_crossing_rate(audio_path):
    """Extract Zero-Crossing Rate (ZCR) from audio file."""
    try:
        # Load audio file
        sound = parselmouth.Sound(audio_path)

        # Get audio samples
        samples = sound.values

        # Calculate zero-crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(samples)))
        zcr = zero_crossings / (2 * len(samples))  # Normalize by signal length

        # Calculate ZCR for different segments
        segment_length = len(samples) // 10  # Divide into 10 segments
        zcr_segments = []

        for i in range(10):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(samples))
            segment = samples[start_idx:end_idx]

            if len(segment) > 1:
                segment_zcr = np.sum(
                    np.diff(np.signbit(segment))) / (2 * len(segment))
                zcr_segments.append(segment_zcr)

        return {
            'zcr_overall': zcr,
            'zcr_mean': np.mean(zcr_segments) if zcr_segments else None,
            'zcr_std': np.std(zcr_segments) if zcr_segments else None,
            'zcr_min': np.min(zcr_segments) if zcr_segments else None,
            'zcr_max': np.max(zcr_segments) if zcr_segments else None
        }

    except Exception as e:
        return {
            'zcr_overall': None,
            'zcr_mean': None,
            'zcr_std': None,
            'zcr_min': None,
            'zcr_max': None
        }


def extract_voice_breaks(audio_path):
    """Extract Voice Breaks / Unvoiced Segments information from audio file."""
    try:
        # Load audio file
        sound = parselmouth.Sound(audio_path)

        # Extract pitch
        pitch = sound.to_pitch()

        # Get pitch values
        pitch_values = pitch.selected_array['frequency']

        # Calculate voiced vs unvoiced segments
        voiced_mask = pitch_values > 0
        unvoiced_mask = pitch_values == 0

        total_frames = len(pitch_values)
        voiced_frames = np.sum(voiced_mask)
        unvoiced_frames = np.sum(unvoiced_mask)

        # Calculate percentages
        voiced_percentage = (voiced_frames / total_frames) * 100
        unvoiced_percentage = (unvoiced_frames / total_frames) * 100

        # Calculate voice breaks (transitions from voiced to unvoiced)
        voice_breaks = np.sum(np.diff(voiced_mask.astype(int)) == -1)

        # Calculate average duration of voiced and unvoiced segments
        voiced_segments = []
        unvoiced_segments = []

        current_segment_length = 1
        current_is_voiced = voiced_mask[0]

        for i in range(1, len(voiced_mask)):
            if voiced_mask[i] == current_is_voiced:
                current_segment_length += 1
            else:
                if current_is_voiced:
                    voiced_segments.append(current_segment_length)
                else:
                    unvoiced_segments.append(current_segment_length)
                current_segment_length = 1
                current_is_voiced = voiced_mask[i]

        # Add the last segment
        if current_is_voiced:
            voiced_segments.append(current_segment_length)
        else:
            unvoiced_segments.append(current_segment_length)

        # Calculate statistics
        avg_voiced_duration = np.mean(
            voiced_segments) if voiced_segments else 0
        avg_unvoiced_duration = np.mean(
            unvoiced_segments) if unvoiced_segments else 0

        return {
            'voice_breaks_count': voice_breaks,
            'voiced_percentage': voiced_percentage,
            'unvoiced_percentage': unvoiced_percentage,
            'avg_voiced_duration': avg_voiced_duration,
            'avg_unvoiced_duration': avg_unvoiced_duration,
            'voiced_segments_count': len(voiced_segments),
            'unvoiced_segments_count': len(unvoiced_segments)
        }

    except Exception as e:
        return {
            'voice_breaks_count': None,
            'voiced_percentage': None,
            'unvoiced_percentage': None,
            'avg_voiced_duration': None,
            'avg_unvoiced_duration': None,
            'voiced_segments_count': None,
            'unvoiced_segments_count': None
        }
