import os, random, copy, json, re, h5py, yaml, pickle, argparse, gc, time
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
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

import sys
current_dir = os.path.dirname(__file__)
raven_dir = os.path.abspath(os.path.join(current_dir, "raven"))
if raven_dir not in sys.path:
    sys.path.insert(0, raven_dir)

from raven.finetune_learner import Learner
from raven.metrics import WER


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

def make_json(split, learner, cfg):
	lipreading_preprocessing_func = get_preprocessing_pipelines()["test"] # We always use test preprocessing for json creation
	processed_count = 0

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
		elif cfg.dataset.name == "facestar":
			if split == "train":
				sub_folder = "train"
			else: ## FaceStar has no validation set
				sub_folder = "test"
	else:
		sub_folder = "pretrain"
	if not cfg.output_file_name:
		fn = cfg.dataset.name + "_" + split + "_" + cfg.model_name + "_" + cfg.visual_corruption.occ_type + ".json"
	else:
		fn = cfg.dataset.name + "_" + split + "_" + cfg.output_file_name + ".json"
	output_file = os.path.join(cfg.output_file_path, fn)
	os.makedirs(os.path.dirname(output_file), exist_ok=True)

	jsn = []
	if cfg.resume:
		assert os.path.exists(output_file)
		with open(output_file, 'r') as json_file:
			jsn = json.load(json_file)
			uid_keys = set([item.get("Uid") for item in jsn if item.get("Uid") and item.get("nhyps")])
			print(f"Resume -> skip {len(uid_keys)} UIDs")

	with open(os.path.join(cfg.original_dataset_path, split + '.txt'), 'r') as txt_file:
		total_lines = sum(1 for _ in txt_file)

	with open(os.path.join(cfg.original_dataset_path, split + '.txt'), 'r') as txt_file:
		# noise_list = load_noise_list(noise_path, noise)
		np.random.seed(cfg.hyperparameters.seed)

		for line in tqdm(txt_file, total = total_lines, desc="Processing meta data", dynamic_ncols=True):
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
			meta_data["Clean_Wav"] = "" # no need
			meta_data["Noise_Wav"] = "" # no need
			meta_data["Noise_Category"] = cfg.visual_corruption.occ_type
			meta_data["SNR"] = 0 # no need
			meta_data["Mouthroi"] = os.path.join(cfg.cropped_hdf5_path, sub_folder, line + '_crop.hdf5')
			meta_data["Video"] = os.path.join(cfg.cropped_hdf5_path, sub_folder, line + '_crop.mp4')
			meta_data["Face_landmark"] = os.path.join(cfg.landmark_path, sub_folder, line + '.pkl')

			if not (os.path.exists(meta_data["Mouthroi"]) and os.path.exists(meta_data["Face_landmark"])):
				print(f'Mouthroi or Face landmark not exists: {sub_folder}/{line} -> skip')
				continue

			try:
				hyps, scores, occlude_cfg = load_nhpys(learner,
										   		  cfg,
										   		  visual_corruption_model=visual_corruption_model,
												  video_path=meta_data["Mouthroi"],
												  landmark_path=meta_data["Face_landmark"],
												  lipreading_preprocessing_func=lipreading_preprocessing_func,
												  beam_size=cfg.hyperparameters.BEAM_SIZE,
												  n_hyp=cfg.hyperparameters.N_HYP,
												  )
				meta_data["nhyps"] = {"hyps": hyps, "scores": scores}
				meta_data["Visual_Corruption"] = occlude_cfg
				meta_data["WER_1st-hyp"] = round(calculate_wer([meta_data["nhyps"]["hyps"][0]], [meta_data["Caption"]]), 2)
			except RuntimeError:
				print(meta_data["Mouthroi"])
				torch.cuda.empty_cache()
				gc.collect()
				time.sleep(10)
				continue

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

@torch.no_grad()
def load_nhpys(learner,
			   cfg,
			   visual_corruption_model,
			   video_path,
			   landmark_path=None, 
			   lipreading_preprocessing_func=None, 
			   beam_size=50,
			   n_hyp = 5):
	
	video = load_mouthroi(video_path)
	if video.shape[0] > cfg.hyperparameters.max_video_length:
		# video = video[:cfg.hyperparameters.max_video_length]
		raise RuntimeError
	if visual_corruption_model is not None:
		video, occlude_config = add_noise(video, landmark_path, visual_corruption_model, cfg)
	else:
		occlude_config = {}

	video = lipreading_preprocessing_func(torch.from_numpy(video).cuda()).float()
	results, scores = learner.get_nbest_hyps(video)
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
	
	return input, score, occlude_config

def add_noise(video, lm_path, visual_corruption_model, cfg):
	with open(lm_path, "rb") as pkl_file:
		pkl = pickle.load(pkl_file)
	lm = pkl['landmarks']
	yx_min = pkl['yx_min']

	# occlusion -> coco, hands, pixelate, blur
	video, _, occlude_config = visual_corruption_model.occlude_sequence(video, lm, yx_min, fixlen=cfg.visual_corruption.noise_chunk_fixlen, freq=1, return_config=True)
	return video, occlude_config

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


@hydra.main(config_path="raven/conf", config_name="config_test")
def main(cfg):
	cfg.gpus = torch.cuda.device_count()
	print("num gpus:", cfg.gpus)
	assert cfg.gpus == 1

	vsr_config_path = cfg.get("vsr_config", "conf/vsr_config.yaml")
	vsr_config_path = os.path.join(os.path.dirname(__file__), vsr_config_path)
	vsr_cfg = OmegaConf.load(vsr_config_path)

	print("VSR config:", vsr_cfg)

	# Learner -> raven learner
	learner = Learner(cfg).cuda()
	learner.eval()

	for splt in vsr_cfg.dataset.split:
		make_json(splt, learner, vsr_cfg)


# Run: python make_json_vsr.py +vsr_config=<path_to_vsr_config.yaml>
if __name__ == "__main__":	
	main()

