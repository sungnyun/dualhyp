import os, random, copy, json, re, yaml, argparse
import numpy as np
import torch
import torchaudio
import editdistance
from tqdm import tqdm
from num2words import num2words
from omegaconf import OmegaConf
from evaluate import load  ## pip install jiwer
eval_wer = load("wer")

import whisper
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()


def make_json(split, model, cfg, shard_index=0, num_shards=1):
	processed_count = 0
	if cfg.audio_corruption.enabled:
		noise = cfg.audio_corruption.noise_type
	else:
		noise = "none"

	if split in ["train", "val", "test"]:
		if cfg.dataset.name == "LRS2":
			sub_folder = "main"
		elif cfg.dataset.name == "LRS3":
			if split in ["train", "val"]:
				sub_folder = "trainval"
			else:
				sub_folder = "test"
		elif cfg.dataset.name == "facestar":
			if split == "train":
				sub_folder = "train"
			else: ## FaceStar has no validation set
				sub_folder = "test"
	else:
		sub_folder = "pretrain"
	if not cfg.output_file_name:
		fn = cfg.dataset.name + "_" + split + "_" + cfg.model_name + "_" + cfg.audio_corruption.noise_type + ".json"
	else:
		fn = cfg.dataset.name + "_" + split + "_" + cfg.output_file_name + ".json"
	if num_shards > 1:
		fn = fn[:-5] + f"_{shard_index:02d}.json"
	output_file = os.path.join(cfg.output_file_path, fn)
	os.makedirs(os.path.dirname(output_file), exist_ok=True)

	jsn = []
	if cfg.resume:
		assert os.path.exists(output_file)
		with open(output_file, 'r') as json_file:
			jsn = json.load(json_file)
			uid_keys = set([item.get("Uid") for item in jsn if item.get("Uid")])
			print(f"Resume -> skip {len(uid_keys)} UIDs")

	with open(os.path.join(cfg.original_dataset_path, split + '.txt'), 'r') as txt_file:
		all_lines = [line.strip() for line in txt_file]
		total_lines = len(all_lines)

	# Shard the lines
	shard_size = (total_lines + num_shards - 1) // num_shards  # Divide evenly
	start_idx = shard_index * shard_size
	end_idx = min(start_idx + shard_size, total_lines)
	lines_to_process = all_lines[start_idx:end_idx]

	print(f"Processing shard {shard_index + 1}/{num_shards}: {len(lines_to_process)} lines")

	# with open(os.path.join(cfg.original_dataset_path, split + '.txt'), 'r') as txt_file:
	noise_list = load_noise_list(cfg.noise_path, noise, split) if noise != "none" else None
	np.random.seed(cfg.hyperparameters.seed)

	for line in tqdm(lines_to_process, total = len(lines_to_process), desc="Processing meta data", dynamic_ncols=True):
		line = line.strip()
		if split == "test":
			line = line.split(" ")[0]
		if cfg.resume:
			if line in uid_keys:
				continue
		meta_data = dict.fromkeys(cfg.json_keys)
		meta_data["Dataset"] = cfg.dataset.name
		meta_data["Uid"] = line
		meta_data["Caption"] = load_caption(os.path.join(cfg.original_dataset_path, sub_folder, line))
		meta_data["Clean_Wav"] = os.path.join(cfg.extracted_audio_path, sub_folder, line + '.wav')
		meta_data["Noise_Wav"] = noise_list[np.random.randint(0, len(noise_list))] if noise_list is not None else None
		meta_data["Noise_Category"] = noise
		if cfg.audio_corruption.noise_chunk:
			low, high = cfg.audio_corruption.noise_snr_heavy
		else:
			low, high = cfg.audio_corruption.noise_snr
		meta_data["SNR"] = np.random.randint(low, high+1)
		meta_data["Mouthroi"] = "" # no need
		meta_data["Video"] = "" # no need
		meta_data["Face_landmark"] = "" # no need

		if not os.path.exists(meta_data["Clean_Wav"]):
			print(f'Audio file not exists: {sub_folder}/{line} -> skip')
			continue

		try:
			hyps, scores, noise_cfg = load_nhpys(model,
										cfg,
										audio_path=meta_data["Clean_Wav"],
										noise_path=meta_data["Noise_Wav"],
										snr=meta_data["SNR"],
										n_mel=cfg.hyperparameters.n_mel,
										beam_size=cfg.hyperparameters.BEAM_SIZE,
										n_hyp=cfg.hyperparameters.N_HYP,
										)
			meta_data["nhyps"] = {"hyps": hyps, "scores": scores}
			meta_data["Audio_Corruption"] = noise_cfg
			meta_data["WER_1st-hyp"] = round(calculate_wer([meta_data["nhyps"]["hyps"][0]], [meta_data["Caption"]]), 2)
		except RuntimeError:
			print (meta_data["Clean_Wav"])
			continue
			# print (meta_data["Noise_Wav"])

		jsn.append(meta_data)

		processed_count += 1
		if processed_count % cfg.hyperparameters.save_interval == 0:
			with open(output_file, 'w') as json_file:
				json.dump(jsn, json_file, indent=4)  # indent=4 makes the JSON readable
				# torch.cuda.empty_cache()

	## Dump json file
	with open(output_file, 'w') as json_file:
		json.dump(jsn, json_file, indent=4)  # indent=4 makes the JSON readable
	print(f"JSON file '{output_file}' has been created.")


def load_caption(line):
	caption_path = line + ".txt"
	with open(caption_path) as caption_file:
		gt = ' '.join(caption_file.readline().strip().split()[1:])
		caption = normalize(gt, gt_or_hyp = "output")

	return caption

def load_noise_list(noise_path, noise_category, split):
	assert noise_category in {"all", "babble", "speech", "music", "noise"}
	if split in ['train', 'pretrain']:
		noise_fn = os.path.join(noise_path, "tsv", noise_category, "train.tsv")
		if noise_category == "speech":
			noise_fn = os.path.join(noise_path, noise_category, "train.tsv")
	elif split == 'val':
		noise_fn = os.path.join(noise_path, "tsv", noise_category, "valid.tsv")
		if noise_category == "speech":
			noise_fn = os.path.join(noise_path, noise_category, "valid.tsv")
	elif split == 'test':
		noise_fn = os.path.join(noise_path, "tsv", noise_category, "test.tsv")
		if noise_category == "speech":
			noise_fn = os.path.join(noise_path, noise_category, "test.tsv")	
	else:
		raise NotImplementedError(f"Unknown split: {split}")

	noise_list = []
	with open(noise_fn, "r") as f:
		for ln in f:
			noise_list.append(ln.strip())
	return noise_list

@torch.no_grad()
def load_nhpys(model, 
			   cfg,
			   audio_path, 
			   noise_path=None, 
			   snr=0,			   
			   n_mel=128,
			   beam_size=50,
			   n_hyp = 5):
	## Load audio and add noise with given snr level
	audio = whisper.load_audio(audio_path)
	if audio.shape[0] > cfg.hyperparameters.max_audio_length:
		# audio = audio[:cfg.hyperparameters.max_audio_length]
		raise RuntimeError
	if cfg.audio_corruption.enabled:
		noise = whisper.load_audio(noise_path)
		audio, noise_cfg = add_noise(audio, noise, snr, cfg)
		noise_cfg["noise_name"] = os.path.join(os.path.basename(os.path.dirname(noise_path)), os.path.basename(noise_path))
	else:
		noise_cfg = {}
	audio = whisper.pad_or_trim(audio)

	mel = whisper.log_mel_spectrogram(audio, n_mels = n_mel).to(model.device)
	options = whisper.DecodingOptions(language='en', beam_size=beam_size, fp16=False,)
	result_dict = whisper.decode(model, mel, options)
	results = result_dict.texts
	confidences = result_dict.avg_logprob
	
	for i in range(len(results)):
		text = normalize(results[i])
		results[i] = text if len(text) > 0 else '<UNK>'

	input, score = [], []
	for (result, confidence) in zip(results, confidences):
		if len(input) < n_hyp and len(result) > 0 and result not in input: ## Repetitives are erased.
			input.append(result)
			score.append(confidence)

	if len(input) < n_hyp:
		input_len = len(input)
		for _ in range(n_hyp - input_len):
			idx = random.choice(range(input_len))
			repeat_input = copy.deepcopy(input[idx])
			repeat_score = copy.deepcopy(score[idx])

			input.append(repeat_input)
			score.append(repeat_score)
	
	return input, score, noise_cfg

def add_noise(audio, noise, snr, cfg):
	audio_rms = np.sqrt(np.mean(np.square(audio), axis = -1))
	if len(audio) >= len(noise):
		ratio = int(np.ceil(len(audio) / len(noise)))
		noise = np.concatenate([noise for _ in range(ratio)])
	if len(audio) < len(noise):
		start = 0
		noise = noise[start : start + len(audio)]
	noise_rms = np.sqrt(np.mean(np.square(noise), axis=-1))
	adjusted_noise_rms = audio_rms / (10**(snr / 20))
	adjusted_noise = noise * (adjusted_noise_rms / noise_rms)

	if cfg.audio_corruption.noise_chunk:
		if cfg.audio_corruption.noise_chunk_fixlen:
			occ_len = int(len(audio) * cfg.audio_corruption.noise_chunk_fixlen)
		else:
			occ_len = int(len(audio) * np.random.beta(2, 2, size=1)[0])
		start_fr = np.random.randint(0, len(audio) - occ_len)
		mixed = audio
		mixed[start_fr:start_fr+occ_len] += adjusted_noise[start_fr:start_fr+occ_len]
	else:
		mixed = audio + adjusted_noise
		start_fr, occ_len = 0, len(mixed)

	noise_cfg = {
		'total_len': len(audio),
		"start_fr": start_fr,
		"occ_len": occ_len,
		"snr": snr,
	}
	return mixed, noise_cfg

def normalize(string, gt_or_hyp = "input"):
	try:
		output = normalizer(string)
		output = re.sub(r"[-+]?\d*\.?\d+|\d+%?", lambda m: num2words(m.group()), output).replace('%', ' percent')
	except Exception:
		output = normalizer(string)
		print(f'{gt_or_hyp} exception: {output}')

	return output

def calculate_wer(all_hypo, all_refer):
    return eval_wer.compute(predictions=all_hypo, references=all_refer)


# Run: python make_json_asr.py --config=<path_to_asr_config.yaml>
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run make_json_asr with a YAML config file")
	parser.add_argument("--config", type=str, default="conf/asr_config.yaml", help="Path to YAML configuration file")
	parser.add_argument("--shard_index", type=int, default=0, help="Index of the current shard (0-based)")
	parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
	args = parser.parse_args()
    
    # Load the YAML configuration file
	config = OmegaConf.load(args.config)

	config.gpus = torch.cuda.device_count()
	print("num gpus:", config.gpus)
	assert config.gpus == 1

	if config.model_name == 'whisper-large':
		model = whisper.load_model('large').cuda()
		config.hyperparameters.n_mel = 128   # 128 for large, 80 for base
	else:
		raise NotImplementedError

	for splt in config.dataset.split:
		make_json(splt, model, config, shard_index=args.shard_index, num_shards=args.num_shards)

