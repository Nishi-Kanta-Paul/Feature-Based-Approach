import os
import pandas as pd
from utils import find_all_audio_paths, extract_shimmer

CSV_PATH = "all_audios_mapped_id_for_label/final_selected.csv"
AUDIO_BASE = "Processed_data_sample_raw_voice/raw_wav"
OUTPUT_PATH = "features/shimmer_features.csv"
LOG_PATH = "features/shimmer_extraction_errors.log"

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
    print("ERROR: 'audio_audio.m4a' column not found. Please check your CSV header and delimiter.")
    exit(1)

os.makedirs("features", exist_ok=True)
error_log = open(LOG_PATH, "w")

# Statistics tracking
stats = {
    "total_files": 0,
    "files_found": 0,
    "files_not_found": 0,
    "successful_extractions": 0,
    "failed_extractions": 0,
    "too_short": 0,
    "silent": 0
}

results = []
for idx, row in df.iterrows():
    audio_id = str(row['audio_audio.m4a'])
    row_id = row['ROW_ID']
    stats["total_files"] += 1

    audio_paths = find_all_audio_paths(AUDIO_BASE, audio_id)
    if audio_paths:
        stats["files_found"] += 1
        for audio_path in audio_paths:
            print(f"Processing: {audio_path}")
            shimmer, status = extract_shimmer(audio_path)

            # Update statistics
            if status.startswith("success"):
                stats["successful_extractions"] += 1
                print(f"  ✓ Success: {shimmer:.6f} ({status})")
            elif status == "too_short":
                stats["too_short"] += 1
            elif status == "silent":
                stats["silent"] += 1
            else:
                stats["failed_extractions"] += 1
                print(f"  ✗ Failed: {status}")

            if not status.startswith("success"):
                error_log.write(f"{audio_path}: {status}\n")

            results.append({
                "ROW_ID": row_id,
                "audio_id": audio_id,
                "wav_filename": os.path.basename(audio_path),
                "shimmer": shimmer,
                "status": status
            })
    else:
        stats["files_not_found"] += 1
        error_log.write(f"{audio_id}: file_not_found\n")
        results.append({
            "ROW_ID": row_id,
            "audio_id": audio_id,
            "wav_filename": None,
            "shimmer": None,
            "status": "file_not_found"
        })

error_log.close()

# Print statistics
print("\n=== SHIMMER EXTRACTION STATISTICS ===")
print(f"Total files processed: {stats['total_files']}")
print(f"Files found: {stats['files_found']}")
print(f"Files not found: {stats['files_not_found']}")
print(f"Successful extractions: {stats['successful_extractions']}")
print(f"Failed extractions: {stats['failed_extractions']}")
print(f"Too short (<0.1s): {stats['too_short']}")
print(f"Silent files: {stats['silent']}")
if stats['files_found'] > 0:
    success_rate = stats['successful_extractions'] / stats['files_found'] * 100
    print(f"Success rate: {success_rate:.1f}%")
else:
    print("Success rate: 0%")

# Save results
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to: {OUTPUT_PATH}")
print(f"Error log saved to: {LOG_PATH}")

# Show some successful extractions
successful_results = result_df[result_df['status'].str.startswith(
    'success', na=False)]
if not successful_results.empty:
    print(f"\nSample successful shimmer extractions:")
    print(successful_results[['audio_id', 'shimmer', 'status']].head(10))
