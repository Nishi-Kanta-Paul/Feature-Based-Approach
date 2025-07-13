import os
import pandas as pd
from utils import find_all_audio_paths, extract_zero_crossing_rate

CSV_PATH = "all_audios_mapped_id_for_label/final_selected.csv"
AUDIO_BASE = "Processed_data_sample_raw_voice/raw_wav"
OUTPUT_PATH = "features/zcr_features.csv"
LOG_PATH = "features/zcr_extraction_errors.log"

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
    error_log.write("ZCR Extraction Errors Log\n")
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
            # Extract ZCR features
            zcr_features = extract_zero_crossing_rate(audio_path)

            # Check if extraction was successful
            if zcr_features['zcr_overall'] is not None:
                result = {
                    'audio_id': audio_id,
                    'audio_path': audio_path,
                    **zcr_features
                }
                results.append(result)
                success_count += 1
                print(f"✅ Successfully extracted ZCR features for {audio_id}")
                print(f"   Overall ZCR: {zcr_features['zcr_overall']:.4f}")
                print(f"   Mean ZCR: {zcr_features['zcr_mean']:.4f}")
                print(f"   ZCR Std: {zcr_features['zcr_std']:.4f}")
            else:
                error_msg = f"ZCR extraction failed - no valid signal found"
                print(f"❌ {error_msg}")
                error_log.write(f"{audio_id}: {error_msg}\n")
                error_count += 1

        except Exception as e:
            error_msg = f"Error extracting ZCR: {str(e)}"
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
    print(f"\n📊 ZCR Extraction Summary:")
    print(f"   Total files processed: {len(audio_ids)}")
    print(f"   Files found: {len(audio_paths)}")
    print(f"   Successful extractions: {success_count}")
    print(f"   Failed extractions: {error_count}")
    print(f"   Success rate: {(success_count/len(audio_ids)*100):.1f}%")

    # Print sample results
    print(f"\n📈 Sample ZCR Statistics:")
    successful_results = results_df[results_df['zcr_overall'].notna()]
    if not successful_results.empty:
        print(
            f"   Average Overall ZCR: {successful_results['zcr_overall'].mean():.4f}")
        print(
            f"   Average Mean ZCR: {successful_results['zcr_mean'].mean():.4f}")
        print(
            f"   Average ZCR Std: {successful_results['zcr_std'].mean():.4f}")
        print(
            f"   ZCR Range: {successful_results['zcr_overall'].min():.4f} - {successful_results['zcr_overall'].max():.4f}")

        print(f"\n🎯 Top 5 ZCR Values (Highest):")
        top_zcr = successful_results.nlargest(5, 'zcr_overall')[
            ['audio_id', 'zcr_overall', 'zcr_mean']]
        for _, row in top_zcr.iterrows():
            print(
                f"   {row['audio_id']}: Overall={row['zcr_overall']:.4f}, Mean={row['zcr_mean']:.4f}")

        print(f"\n🎯 Top 5 ZCR Values (Lowest):")
        bottom_zcr = successful_results.nsmallest(
            5, 'zcr_overall')[['audio_id', 'zcr_overall', 'zcr_mean']]
        for _, row in bottom_zcr.iterrows():
            print(
                f"   {row['audio_id']}: Overall={row['zcr_overall']:.4f}, Mean={row['zcr_mean']:.4f}")
else:
    print("❌ No ZCR features were successfully extracted!")
    error_count = len(audio_ids)

print(f"\n📝 Error log saved to {LOG_PATH}")
print(f"🔍 Check the error log for detailed failure reasons")
