import json
import sys
import os

VSR_KEYS = ["Mouthroi", "Video", "Face_landmark", "Visual_Corruption"]

def merge_json_files(asr_filename, vsr_filename, merged_filename):
    # Load the lists of dictionaries from each file.
    with open(asr_filename, "r") as f:
        asr_list = json.load(f)
    with open(vsr_filename, "r") as f:
        vsr_list = json.load(f)

    asr_dict = {item.get("Uid"): item for item in asr_list if item.get("Uid")}
    vsr_dict = {item.get("Uid"): item for item in vsr_list if item.get("Uid")}

    merged_list = []
    unprocessed = []
    
    # Iterate over Uids that appear in both files.
    common_uids = set(asr_dict.keys()) & set(vsr_dict.keys())
    for uid in common_uids:
        asr_item = asr_dict[uid]
        vsr_item = vsr_dict[uid]

        # Check that both items have a non-null (and non-empty) "nhyps" field.
        if not asr_item.get("nhyps") or not vsr_item.get("nhyps"):
            unprocessed.append(uid)
            continue

        merged_item = {}

        # Merge all keys from the ASR item.
        # Renamed "nhyps" to "nhyps_asr".
        for key, value in asr_item.items():
            if key == "nhyps":
                merged_item["nhyps_asr"] = value
            else:
                merged_item[key] = value

        # Merge all keys from the VSR item.
        # Renamed "nhyps" to "nhyps_vsr".
        for key, value in vsr_item.items():
            if key == "nhyps":
                merged_item["nhyps_vsr"] = value
            elif key == "Noise_Category":
                merged_item["Noise_Category"] = (asr_item.get("Noise_Category"), value)
            elif key == "WER_1st-hyp":
                merged_item["WER_1st-hyp"] = (asr_item.get("WER_1st-hyp"), value)
            elif key in VSR_KEYS:
                merged_item[key] = value

        merged_list.append(merged_item)

    # Write the merged list to the output JSON file.
    # Check if the merged list is empty.
    if os.path.exists(merged_filename):
        print(f"File {merged_filename} already exists. Merging into the existing file.")
        sys.exit(1)
    with open(merged_filename, "w") as f:
        json.dump(merged_list, f, indent=4)
    print(f"Merged JSON file saved to {merged_filename}")
    print(f"Unprocessed Uids: {unprocessed}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_json.py <asr_file_path> <vsr_file_path> <merged_output_path>")
        sys.exit(1)
        
    asr_file = sys.argv[1]
    vsr_file = sys.argv[2]
    merged_file = sys.argv[3]

    merge_json_files(asr_file, vsr_file, merged_file)