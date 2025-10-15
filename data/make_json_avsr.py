import os, random, copy, json, re, h5py, yaml, pickle, argparse
import hydra
import shutil, tempfile, subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from num2words import num2words
from evaluate import load  ## pip install jiwer
from omegaconf import OmegaConf
eval_wer = load("wer")

import torch
import torchaudio
from utils import Compose, Normalize, RandomCrop, HorizontalFlip, CenterCrop
from visual_corruption import *

import whisper
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

import sys
current_dir = os.path.dirname(__file__)
auto_avsr_dir = os.path.abspath(os.path.join(current_dir, "auto_avsr"))
if auto_avsr_dir not in sys.path:
    sys.path.insert(0, auto_avsr_dir)
from auto_avsr.lightning_av import ModelModule


def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([
                                Normalize( 0.0,255.0 ),
                                RandomCrop(crop_size),
                                # HorizontalFlip(0.5),
                                Normalize(mean, std) ])
    preprocessing['val'] = Compose([
                                Normalize( 0.0,255.0 ),
                                CenterCrop(crop_size),
                                Normalize(mean, std) ])
    preprocessing['test'] = preprocessing['val']
    return preprocessing

def load_mouthroi(filename):
    with h5py.File(filename, 'r') as hf:
        return hf['video_frames'][:]

def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def make_json(split, learner, cfg, asr_data):
	lipreading_preprocessing_func = get_preprocessing_pipelines()["test"] # We always use test preprocessing for json creation
	processed_count = 0
	if cfg.audio_corruption.enabled:
		noise = cfg.audio_corruption.noise_type
	else:
		noise = "none"
	if cfg.visual_corruption.enabled:
		visual_corruption_model = Visual_Corruption_Modeling(
			occlusion_patch_dir=cfg.occlusion_patch_dir,
			occ_type=cfg.visual_corruption.occ_type
		)
	else:
		visual_corruption_model = None
		cfg.visual_corruption.occ_type = "none"

	if split in ["train", "val", "test"]:
		if cfg.dataset.name == "LRS2":
			sub_folder = "main"
		elif cfg.dataset.name == "LRS3":
			if split in ["train", "val"]:
				sub_folder = "trainval"
			else:
				sub_folder = "test"
	else:
		sub_folder = "pretrain"
	if not cfg.output_file_name:
		fn = cfg.dataset.name + "_" + split + "_" + cfg.model_name + "_" + cfg.audio_corruption.noise_type + "_" + cfg.visual_corruption.occ_type + ".json"
	else:
		fn = cfg.dataset.name + "_" + split + "_" + cfg.output_file_name + ".json"
	output_file = os.path.join(cfg.output_file_path, fn)
	os.makedirs(os.path.dirname(output_file), exist_ok=True)

	uid2asr = {}
	for asr_dict in asr_data:
		uid2asr[asr_dict["Uid"]] = asr_dict

	jsn = []
	if cfg.resume:
		assert os.path.exists(output_file)
		with open(output_file, 'r') as json_file:
			jsn = json.load(json_file)
			uid_keys = set([item.get("Uid") for item in jsn if item.get("Uid")])
			print(f"Resume -> skip {len(uid_keys)} UIDs")

	with open(os.path.join(cfg.original_dataset_path, split + '.txt'), 'r') as txt_file:
		total_lines = sum(1 for _ in txt_file)

	with open(os.path.join(cfg.original_dataset_path, split + '.txt'), 'r') as txt_file:
		# noise_list = load_noise_list(cfg.noise_path, noise, split) if noise != "none" else None
		np.random.seed(cfg.hyperparameters.seed)

		for line in tqdm(txt_file, total = total_lines, desc="Processing meta data", dynamic_ncols=True):
			line = line.strip()
			if split == "test":
				line = line.split(" ")[0]
			if cfg.resume:
				if line in uid_keys:
					continue
			if line not in uid2asr:
				print(f"Warning: {line} not in ASR data -> skip")
				continue
			meta_data = dict.fromkeys(cfg.json_keys)
			meta_data["Dataset"] = cfg.dataset.name
			meta_data["Uid"] = line
			meta_data["Caption"] = load_caption(os.path.join(cfg.original_dataset_path, sub_folder, line))
			meta_data["Clean_Wav"] = uid2asr[line]["Clean_Wav"]
			meta_data["Noise_Wav"] = uid2asr[line]["Noise_Wav"]
			meta_data["Audio_Noise_Category"] = uid2asr[line]["Noise_Category"][0]
			meta_data["Noise_Category"] = uid2asr[line]["Noise_Category"][1]
			meta_data["SNR"] = uid2asr[line]["SNR"]
			meta_data["Mouthroi"] = os.path.join(cfg.cropped_hdf5_path, sub_folder, line + '_crop.hdf5')
			meta_data["Video"] = os.path.join(cfg.cropped_hdf5_path, sub_folder, line + '_crop.mp4')
			meta_data["Face_landmark"] = os.path.join(cfg.landmark_path, sub_folder, line + '.pkl')

			if not (os.path.exists(meta_data["Mouthroi"]) and os.path.exists(meta_data["Face_landmark"])):
				print(f'Mouthroi or Face landmark not exists: {sub_folder}/{line} -> skip')
				continue

			try:
				hyps, scores, noise_cfg, occlude_cfg = load_nhpys(learner,
										   		  cfg,
										   		  visual_corruption_model=visual_corruption_model,
												  asr_cfg=uid2asr[line],
												  video_path=meta_data["Mouthroi"],
												  landmark_path=meta_data["Face_landmark"],
												  lipreading_preprocessing_func=lipreading_preprocessing_func,
												  beam_size=cfg.hyperparameters.BEAM_SIZE,
												  n_hyp=cfg.hyperparameters.N_HYP,
												  )
				meta_data["nhyps"] = {"hyps": hyps, "scores": scores}
				meta_data["Audio_Corruption"] = noise_cfg
				meta_data["Visual_Corruption"] = occlude_cfg
				meta_data["WER_1st-hyp"] = round(calculate_wer([meta_data["nhyps"]["hyps"][0]], [meta_data["Caption"]]), 2)
			except RuntimeError:
				print(meta_data["Mouthroi"])

			jsn.append(meta_data)

			processed_count += 1
			if processed_count % cfg.hyperparameters.save_interval == 0:
				with open(output_file, 'w') as json_file:
					json.dump(jsn, json_file, indent=4)  # indent=4 makes the JSON readable
					torch.cuda.empty_cache()

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

def load_nhpys(learner,
			   cfg,
			   visual_corruption_model,
			   asr_cfg,
			   video_path,
			   landmark_path=None, 
			   lipreading_preprocessing_func=None, 
			   beam_size=50,
			   n_hyp = 5):
	
	audio = whisper.load_audio(asr_cfg["Clean_Wav"])
	if audio.shape[0] > cfg.hyperparameters.max_audio_length:
		raise RuntimeError
	if cfg.audio_corruption.enabled:
		noise = whisper.load_audio(asr_cfg["Noise_Wav"])
		noise_cfg = asr_cfg["Audio_Corruption"]
		audio = add_audio_noise(audio, noise, noise_cfg)
	else:
		noise_cfg = {}
	audio = torch.from_numpy(audio).float().cuda()
	
	video = load_mouthroi(video_path)
	if video.shape[0] > cfg.hyperparameters.max_video_length:
		raise RuntimeError
	if visual_corruption_model is not None:
		occlude_config = asr_cfg["Visual_Corruption"]
		video = add_noise(video, landmark_path, visual_corruption_model, cfg, occlude_config)
	else:
		occlude_config = {}

	video = lipreading_preprocessing_func(torch.from_numpy(video).cuda()).float()
	audio = cut_or_pad(audio, len(video) * 640, dim=0)

	results, scores = learner.get_nbest_hyps(audio, video)
	results, scores = results[:beam_size], scores[:beam_size]

	for i in range(len(results)):
		text = normalize(results[i])
		results[i] = text if len(text) > 0 else '<UNK>'
	
	input, score = [], []
	for result, sc in zip(results, scores):
		if len(input) < n_hyp and len(result) > 0 and result not in input:
			input.append(result)
			score.append(sc)
		
	if len(input) < n_hyp:
		input_len = len(input)
		for _ in range(n_hyp - input_len):
			idx = random.choice(range(input_len))
			repeat_input = copy.deepcopy(input[idx])
			repeat_score = copy.deepcopy(score[idx])

			input.append(repeat_input)
			score.append(repeat_score)
	
	return input, score, noise_cfg, occlude_config

def add_audio_noise(audio, noise, noise_cfg):
	snr = noise_cfg["snr"]
	start_fr = noise_cfg["start_fr"]
	occ_len = noise_cfg["occ_len"]

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

	# if cfg.audio_corruption.noise_chunk:
	# 	if cfg.audio_corruption.noise_chunk_fixlen:
	# 		occ_len = int(len(audio) * cfg.audio_corruption.noise_chunk_fixlen)
	# 	else:
	# 		occ_len = int(len(audio) * np.random.beta(2, 2, size=1)[0])
		# start_fr = np.random.randint(0, len(audio) - occ_len)
	mixed = audio
	mixed[start_fr:start_fr+occ_len] += adjusted_noise[start_fr:start_fr+occ_len]
	# else:
	# 	mixed = audio + adjusted_noise
	# 	start_fr, occ_len = 0, len(mixed)
	return mixed

def add_noise(video, lm_path, visual_corruption_model, cfg, occlude_config):
	with open(lm_path, "rb") as pkl_file:
		pkl = pickle.load(pkl_file)
	lm = pkl['landmarks']
	yx_min = pkl['yx_min']

	# occlusion -> coco, hands, pixelate, blur
	video, _ = visual_corruption_model.occlude_sequence(video, lm, yx_min, fixlen=cfg.visual_corruption.noise_chunk_fixlen, freq=1, occlude_config=occlude_config)
	return video

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


## for debugging
def write_video_ffmpeg(rois, target_path, ffmpeg='/usr/bin/ffmpeg'):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir+'/%0'+str(decimals)+'d.png' + "'\n")
    ## ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', target_path]
    pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    return


@hydra.main(config_path="auto_avsr/conf", config_name="config")
def main(cfg):
	cfg.gpus = torch.cuda.device_count()
	print("num gpus:", cfg.gpus)
	assert cfg.gpus == 1

	avsr_config_path = cfg.get("avsr_config", "conf/avsr_config.yaml")
	avsr_config_path = os.path.join(os.path.dirname(__file__), avsr_config_path)
	avsr_cfg = OmegaConf.load(avsr_config_path)

	print("AVSR config:", avsr_cfg)

	with open(avsr_cfg.avsr_path, 'r') as f:
		avsr_data = json.load(f)

	# Module -> auto-avsr
	learner = ModelModule(cfg).cuda()
	learner.eval()

	for splt in avsr_cfg.dataset.split:
		if splt not in avsr_cfg.avsr_path:
			print(f"Warning: split {splt} not matching with ASR path! Check the path.")
		make_json(splt, learner, avsr_cfg, avsr_data)


# Run: python make_json_avsr.py +avsr_config=<path_to_avsr_config.yaml>
if __name__ == "__main__":	
	main()

