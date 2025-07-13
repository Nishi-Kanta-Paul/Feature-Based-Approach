import os
import pandas as pd
from utils import find_all_audio_paths, extract_hnr

CSV_PATH = "all_audios_mapped_id_for_label/final_selected.csv"
AUDIO_BASE = "Processed_data_sample_raw_voice/raw_wav"
OUTPUT_PATH = "features/hnr_features.csv"
LOG_PATH = "features/hnr_extraction_errors.log"

# Try reading as tab-separated first
df = pd.read_csv(CSV_PATH, sep='\t')
df.columns = df.columns.str.strip()

# If only one column, try comma-separated
if len(df.columns) == 1:
    print("Detected only one column. Trying comma as delimiter...")
    df = pd.read_csv(CSV_PATH, sep=',')
    df.columns = df.columns.str.strip()

if 'audio_audio.m4a' not in df.columns:
    print('Column names:', df.columns.tolist())
    print("ERROR: 'audio_audio.m4a' column not found!")
    exit(1)

# Extract audio IDs (the column contains just the ID numbers)
audio_ids = df['audio_audio.m4a'].astype(str).tolist()
print(f"Found {len(audio_ids)} audio IDs to process")

# Find audio paths
audio_paths = find_all_audio_paths(AUDIO_BASE, audio_ids)
print(f"Found {len(audio_paths)} audio files")

# Initialize results
results = []
success_count = 0
error_count = 0

# Open error log
with open(LOG_PATH, 'w') as error_log:
    error_log.write("HNR Extraction Errors Log\n")
    error_log.write("=" * 50 + "\n\n")

    # Process each audio file
    for audio_id in audio_ids:
        print(f"\nProcessing audio ID: {audio_id}")

        if audio_id not in audio_paths:
            error_msg = f"Audio file not found for ID: {audio_id}"
            print(f"❌ {error_msg}")
            error_log.write(f"{audio_id}: {error_msg}\n")
            error_count += 1
            continue

        audio_path = audio_paths[audio_id]

        try:
            # Extract HNR features
            hnr_features = extract_hnr(audio_path)

            # Check if extraction was successful (at least one method worked)
            successful_methods = [
                k for k, v in hnr_features.items() if v is not None]

            if successful_methods:
                result = {
                    'audio_id': audio_id,
                    'audio_path': audio_path,
                    **hnr_features
                }
                results.append(result)
                success_count += 1
                print(f"✅ Successfully extracted HNR features for {audio_id}")
                print(
                    f"   Successful methods: {', '.join(successful_methods)}")

                # Print the best HNR value
                best_hnr = None
                for method in ['hnr_manual', 'hnr_autocorr', 'hnr_cepstral']:
                    if hnr_features[method] is not None:
                        best_hnr = hnr_features[method]
                        print(f"   Best HNR ({method}): {best_hnr:.2f} dB")
                        break
            else:
                error_msg = f"HNR extraction failed - no methods succeeded"
                print(f"❌ {error_msg}")
                error_log.write(f"{audio_id}: {error_msg}\n")
                error_count += 1

        except Exception as e:
            error_msg = f"Error extracting HNR: {str(e)}"
            print(f"❌ {error_msg}")
            error_log.write(f"{audio_id}: {error_msg}\n")
            error_count += 1

# Create results DataFrame
if results:
    results_df = pd.DataFrame(results)

    # Add original data
    final_df = df.copy()
    # Convert audio_id to string for proper merging
    final_df['audio_audio.m4a'] = final_df['audio_audio.m4a'].astype(str)
    results_df['audio_id'] = results_df['audio_id'].astype(str)
    final_df = final_df.merge(
        results_df, left_on='audio_audio.m4a', right_on='audio_id', how='left')

    # Save results
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Results saved to {OUTPUT_PATH}")

    # Print summary statistics
    print(f"\n📊 HNR Extraction Summary:")
    print(f"   Total files processed: {len(audio_ids)}")
    print(f"   Files found: {len(audio_paths)}")
    print(f"   Successful extractions: {success_count}")
    print(f"   Failed extractions: {error_count}")
    print(f"   Success rate: {(success_count/len(audio_ids)*100):.1f}%")

    # Print sample results
    print(f"\n📈 Sample HNR Statistics:")
    successful_results = results_df.copy()

    # Count successful methods
    method_counts = {}
    for method in ['hnr_manual', 'hnr_autocorr', 'hnr_cepstral']:
        if method in successful_results.columns:
            method_counts[method] = successful_results[method].notna().sum()

    print(f"   Method success rates:")
    for method, count in method_counts.items():
        print(
            f"     {method}: {count}/{len(successful_results)} ({count/len(successful_results)*100:.1f}%)")

    # Print best HNR values
    print(f"\n🎯 Top 5 HNR Values:")
    best_hnr_values = []
    for _, row in successful_results.iterrows():
        best_hnr = None
        for method in ['hnr_manual', 'hnr_autocorr', 'hnr_cepstral']:
            if method in row and row[method] is not None:
                best_hnr = row[method]
                break
        if best_hnr is not None:
            best_hnr_values.append((row['audio_id'], best_hnr))

    # Sort by HNR value and show top 5
    best_hnr_values.sort(key=lambda x: x[1], reverse=True)
    for audio_id, hnr_value in best_hnr_values[:5]:
        print(f"   {audio_id}: {hnr_value:.2f} dB")
else:
    print("❌ No HNR features were successfully extracted!")
    error_count = len(audio_ids)

print(f"\n📝 Error log saved to {LOG_PATH}")
print(f"🔍 Check the error log for detailed failure reasons")
