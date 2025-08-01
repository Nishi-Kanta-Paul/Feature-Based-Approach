import os
import pandas as pd
from utils import find_all_audio_paths, extract_fundamental_frequency

CSV_PATH = "all_audios_mapped_id_for_label/final_selected.csv"
AUDIO_BASE = "Processed_data_sample_raw_voice/raw_wav"
OUTPUT_PATH = "features/f0_features.csv"
LOG_PATH = "features/f0_extraction_errors.log"

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
    error_log.write("F0 Extraction Errors Log\n")
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
            # Extract F0 features
            f0_features = extract_fundamental_frequency(audio_path)

            # Check if extraction was successful
            if f0_features['f0_mean'] is not None:
                result = {
                    'audio_id': audio_id,
                    'audio_path': audio_path,
                    **f0_features
                }
                results.append(result)
                success_count += 1
                print(f"✅ Successfully extracted F0 features for {audio_id}")
                print(f"   F0 Mean: {f0_features['f0_mean']:.2f} Hz")
                print(f"   F0 Range: {f0_features['f0_range']:.2f} Hz")
            else:
                error_msg = f"F0 extraction failed - no voiced segments found"
                print(f"❌ {error_msg}")
                error_log.write(f"{audio_id}: {error_msg}\n")
                error_count += 1

        except Exception as e:
            error_msg = f"Error extracting F0: {str(e)}"
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
    print(f"\n📊 F0 Extraction Summary:")
    print(f"   Total files processed: {len(audio_ids)}")
    print(f"   Files found: {len(audio_paths)}")
    print(f"   Successful extractions: {success_count}")
    print(f"   Failed extractions: {error_count}")
    print(f"   Success rate: {(success_count/len(audio_ids)*100):.1f}%")

    # Print sample results
    print(f"\n📈 Sample F0 Statistics:")
    successful_results = results_df[results_df['f0_mean'].notna()]
    if not successful_results.empty:
        print(
            f"   Average F0 Mean: {successful_results['f0_mean'].mean():.2f} Hz")
        print(
            f"   Average F0 Range: {successful_results['f0_range'].mean():.2f} Hz")
        print(
            f"   Average F0 Std: {successful_results['f0_std'].mean():.2f} Hz")

        print(f"\n🎯 Top 5 F0 Values:")
        top_f0 = successful_results.nlargest(
            5, 'f0_mean')[['audio_id', 'f0_mean', 'f0_range']]
        for _, row in top_f0.iterrows():
            print(
                f"   {row['audio_id']}: Mean={row['f0_mean']:.2f}Hz, Range={row['f0_range']:.2f}Hz")
else:
    print("❌ No F0 features were successfully extracted!")
    error_count = len(audio_ids)

print(f"\n📝 Error log saved to {LOG_PATH}")
print(f"🔍 Check the error log for detailed failure reasons")
