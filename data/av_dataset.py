import h5py
import json
import subprocess
import random
import pickle
import numpy as np
from copy import deepcopy
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.io import wavfile

import data.whisper as whisper
from data.visual_corruption import *
from data.prompts import get_prompts_format
from data.utils import get_preprocessing_pipelines, load_mouthroi, pad_mouth, random_sample_sequence, word_emb_diff, sent_emb_diff


class AVDataset(Dataset):
    def __init__(self, 
                 split, 
                 json_path,
                 max_input_length=-1,
                 max_nhyps=None,
                 nhyps_key="nhyps_asr",
                 random_sample_nhyps=True,
                 tokenizer=None,
                 occlusion_patch_dir="data/occlusion_patch/",
                 audio_mel=False,
                 audio_pad=True, 
                 audio_corruption_enabled=True,
                 visual_corruption_enabled=True,
                 transform_audio=None, 
                 transform_video=None,
                 maximum_audio_length=320000,
                 maximum_video_length=500,
                 prompts_format="GER",
                 apply_chat_template=False,
                 language=None,
        ):
        """
        Args:
            split (str): One of "train", "val", or "test".
            json_path (str): Path to the JSON file containing dataset info.
            max_input_length (int): Maximum length of input sequences.
            nhyps_key (str): Key for the hypotheses in the JSON file.
            tokenizer (Tokenizer): Tokenizer for encoding text.
            occlusion_patch_dir (str): Directory for occlusion patches.
            audio_corruption_enabled (bool): Enable audio corruption. (config needed)
            visual_corruption_enabled (bool): Enable visual corruption. (config needed)
        """
        self.data = []
        self.data2 = []
        if type(json_path) == str:    
            with open(json_path, 'r') as f:
                self.data = json.load(f)
        elif type(json_path) == list:
            for json_file in json_path:
                if "_pretrain" in json_file:
                    with open(json_file, 'r') as f:
                        self.data2 += json.load(f)
                else:
                    with open(json_file, 'r') as f:
                        self.data += json.load(f)

        self.uid2sample = defaultdict(list)
        self.idx2uid = []
        for data in self.data:
            uid = data["Uid"]
            if uid not in self.uid2sample:
                self.idx2uid.append(uid)
            self.uid2sample[uid].append(data)
        for data in self.data2:
            uid = data["Uid"] + "_pretrain"
            if uid not in self.uid2sample:
                self.idx2uid.append(uid)
            self.uid2sample[uid].append(data)

        self.split = split
        assert self.split in ["train", "val", "test"]
        self.max_input_length = max_input_length
        self.max_nhyps = max_nhyps
        self.nhyps_key = nhyps_key
        self.random_sample_nhyps = random_sample_nhyps
        self.tokenizer = tokenizer
        self.audio_mel = audio_mel
        self.audio_pad = audio_pad
        self.audio_corruption_enabled = audio_corruption_enabled
        self.visual_corruption_enabled = visual_corruption_enabled
        self.transform_audio = transform_audio
        self.transform_video = transform_video
        self.maximum_audio_length = maximum_audio_length  # longer than this will be trimmed
        self.maximum_video_length = maximum_video_length  # longer than this will be trimmed
        self.apply_chat_template = apply_chat_template
        self.language = language
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[self.split]
        if self.visual_corruption_enabled:
            # TODO: make this configurable from json
            self.visual_corruption_models = {}
            for occ_type in ["coco", "hands", "pixelate", "blur"]:
                self.visual_corruption_models[occ_type] = Visual_Corruption_Modeling(
                    occlusion_patch_dir=occlusion_patch_dir,  
                    occ_type=occ_type,
                )
        
        # LLM PROMPTS
        prompts_format = get_prompts_format(prompts_format)
        self.prompt_1, self.prompt_2, self.prompt_3 = prompts_format["prompt_1"], prompts_format["prompt_2"], prompts_format["prompt_3"]
        if self.language is not None:
            self.prompt_1 = self.prompt_1.replace("speech recognition system", f"{self.language} speech recognition system")

        if not hasattr(self.tokenizer, 'eos_token'):
            self.tokenizer.eos_token = "</s>"
            print("WARNING: Tokenizer must have an eos_token attribute -> automatically set to </s>")

    def __len__(self):
        return len(self.uid2sample)

    def __getitem__(self, idx):
        # Get the JSON sample
        uid = self.idx2uid[idx]
        sample = random.choice(self.uid2sample[uid])

        audio = self.load_audio(sample)
        video = self.load_video(sample)
        video = self.lipreading_preprocessing_func(torch.from_numpy(video))

        # Apply optional transforms
        if self.transform_audio:
            audio = self.transform_audio(audio)
        if self.transform_video:
            video = self.transform_video(video)

        # Get prompts, input_ids, and labels
        prompts = self.get_prompt(sample)
        if self.max_input_length > 0:
            prompts["input_ids"] = prompts["input_ids"][:self.max_input_length]
            prompts["labels"] = prompts["labels"][:self.max_input_length]

        # Return audio, video, and some additional information if needed.
        return {
            'audio': audio,
            'video': video,
            "uid": sample.get("Uid", ""),
            "ground_truth": sample.get("Caption", ""),
            **prompts,
        }

    def load_audio(self, sample, n_mel=128):
        audio_file = sample["Clean_Wav"]
        noise_file = sample["Noise_Wav"]
        noise_cfg = sample.get("Audio_Corruption", None)
        audio = whisper.load_audio(audio_file)
        # Audio corruption
        if self.audio_corruption_enabled:
            assert noise_cfg is not None
            noise = whisper.load_audio(noise_file)
            audio = self.add_audio_noise(audio, noise, noise_cfg)
        # Trimming
        if audio.shape[0] > self.maximum_audio_length:
            audio = audio[:self.maximum_audio_length]        
        if self.audio_pad:
            audio = whisper.pad_or_trim(audio)
        if self.audio_mel:
            mel = whisper.log_mel_spectrogram(audio, n_mels=n_mel)
            return mel
        return torch.from_numpy(audio)

    def add_audio_noise(self, audio, noise, noise_cfg):
        audio_rms = np.sqrt(np.mean(np.square(audio), axis=-1))
        if len(audio) >= len(noise):
            ratio = int(np.ceil(len(audio) / len(noise)))
            noise = np.concatenate([noise for _ in range(ratio)])
        if len(audio) < len(noise):
            start = 0
            noise = noise[start : start + len(audio)]
        noise_rms = np.sqrt(np.mean(np.square(noise), axis=-1))
        adjusted_noise_rms = audio_rms / (10**(int(noise_cfg["snr"]) / 20))
        adjusted_noise = noise * (adjusted_noise_rms / noise_rms)

        start_fr = noise_cfg["start_fr"]
        occ_len = noise_cfg["occ_len"]
        mixed = audio
        mixed[start_fr:start_fr+occ_len] += adjusted_noise[start_fr:start_fr+occ_len]
        return mixed

    def load_video(self, sample):
        # Load video from HDF5 file.
        video = load_mouthroi(sample["Mouthroi"])
        occlude_cfg = sample.get("Visual_Corruption", None)
        # Video corruption
        if self.visual_corruption_enabled:
            assert occlude_cfg is not None
            occ_type = sample["Noise_Category"][1]
            lm_path = sample["Face_landmark"]

            with open(lm_path, "rb") as pkl_file:
                pkl = pickle.load(pkl_file)                
            lm = pkl['landmarks']
            yx_min = pkl['yx_min']

            visual_corruption_model = self.visual_corruption_models[occ_type]
            video, _ = visual_corruption_model.occlude_sequence(video, lm, yx_min, freq=1, occlude_config=occlude_cfg, return_config=False)
        if video.shape[0] > self.maximum_video_length:
            video = video[:self.maximum_video_length]
        return video

    def get_prompt(self, sample):
        assert self.tokenizer is not None
        new_dict = {}

        if self.max_nhyps is not None:
            other_hypothesis = sample[self.nhyps_key]["hyps"][1:self.max_nhyps]
        else:
            other_hypothesis = sample[self.nhyps_key]["hyps"][1:]

        if self.random_sample_nhyps:
            other_hypothesis = random_sample_sequence(other_hypothesis, len(other_hypothesis))
        
        final_prompt_no_response = self.prompt_1 + sample[self.nhyps_key]["hyps"][0] + self.prompt_2 + '\n' + ('\n').join(other_hypothesis) + self.prompt_3
        final_prompt = final_prompt_no_response + sample["Caption"] + self.tokenizer.eos_token # </s>

        if self.apply_chat_template:
            # Use chat template for models like phi-3.5
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": final_prompt_no_response},
            ]
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # append <|assistant>| at the end
            )
            answer_ids = self.tokenizer(sample["Caption"], add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
            input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.int64)
            input_ids_no_response = torch.tensor(prompt_ids, dtype=torch.int64)
            labels = torch.tensor([-1] * len(prompt_ids) + answer_ids, dtype=torch.int64)
        else:
            # Use traditional tokenization
            input_ids_no_response = self.tokenizer.encode(final_prompt_no_response)
            input_ids = self.tokenizer.encode(final_prompt)
            labels_temp = input_ids.copy()
            labels = [-1] * len(input_ids_no_response) + labels_temp[len(input_ids_no_response):]
            
            input_ids_no_response = torch.tensor(input_ids_no_response, dtype=torch.int64)
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
            labels = torch.tensor(labels, dtype=torch.int64)

        new_dict['input_ids_no_response'] = input_ids_no_response
        new_dict['input_ids'] = input_ids
        new_dict['labels'] = labels
        new_dict['input'] = final_prompt

        return new_dict

    def collate_fn(self, samples):
        inputs = [item["input"] for item in samples]
        uids = [item["uid"] for item in samples]
        ground_truths = [item["ground_truth"] for item in samples]
        input_ids_list = [item["input_ids"] for item in samples]
        input_ids_no_response_list = [item["input_ids_no_response"] for item in samples]
        labels_list = [item["labels"] for item in samples]
        audio_list = [item["audio"] for item in samples]
        video_list = [item["video"] for item in samples]

        # Pad text sequences
        max_len = max(seq.size(0) for seq in input_ids_list)
        def pad_right(seq, pad_value):
            n = max_len - seq.size(0)
            return torch.cat([seq, torch.full((n,), pad_value, dtype=seq.dtype)])
        
        input_ids = torch.stack([pad_right(seq, 0) for seq in input_ids_list])
        labels = torch.stack([pad_right(seq, -1) for seq in labels_list])

        # Pad audio/video sequences using pad_sequence.
        audio = pad_sequence(audio_list, batch_first=True)
        video_list, _ = pad_mouth(video_list)
        video = [torch.FloatTensor(video).unsqueeze(0) for video in video_list]
        video = torch.stack([x for x in video], dim=0)

        return {
            "input": inputs,
            "uid": uids,
            "ground_truth": ground_truths,
            "input_ids": input_ids,
            "input_ids_no_response": input_ids_no_response_list,
            "labels": labels,
            "audio": audio,
            "video": video,
        }
    
    def get_max_seq_length(self):
        # find out the minimum max_seq_length required during fine-tuning (saves memory!)
        lengths = [len(d["input_ids"]) for d in self.data]
        print(f'mean length = {sum(lengths) / len(lengths)}')
        max_seq_length = max(lengths)
        longest_seq_ix = lengths.index(max_seq_length)
        # support easy override at the top of the file
        return (
            max_seq_length,
            max_seq_length,
            longest_seq_ix,
        )
    
    def _check_audio_length(self):
        self.audio_corruption_enabled = False
        from tqdm import tqdm
        for sample in tqdm(self.data, dynamic_ncols=True):
            audio = self.load_audio(sample)
            audio_length = len(audio) / 16000
            if audio_length > 30:
                print(f"Audio length exceeds 30 seconds for sample {sample['Uid']}: {audio_length} seconds")

    def _check_video_length(self):
        self.visual_corruption_enabled = False
        from tqdm import tqdm
        for sample in tqdm(self.data, dynamic_ncols=True):
            video = self.load_video(sample)
            video_length = len(video) / 25
            if video_length > 30:
                print(f"Video length exceeds 30 seconds for sample {sample['Uid']}: {video_length} seconds")


class DualHypothesesAVDataset(AVDataset):
    def __init__(self, 
                 split, 
                 json_path,
                 prompts_format="DualHyp",
                 **kwargs):
        super().__init__(split, json_path, **kwargs)
    
        ###
        self.nhyps_key_asr = "nhyps_asr"
        self.nhyps_key_vsr = "nhyps_vsr"

        prompts_format = get_prompts_format(prompts_format)
        self.prompt_1, self.prompt_2, self.prompt_3 = prompts_format["prompt_1"], prompts_format["prompt_2"], prompts_format["prompt_3"]
        if self.language is not None:
            self.prompt_1 = self.prompt_1.replace("speech recognition system", f"{self.language} speech recognition system")

    def __getitem__(self, idx):
        # Get the JSON sample
        uid = self.idx2uid[idx]
        sample_1, sample_2 = random.choices(self.uid2sample[uid], k=2)

        audio = self.load_audio(sample_1)
        video = self.load_video(sample_2)
        video = self.lipreading_preprocessing_func(torch.from_numpy(video))

        # Apply optional transforms
        if self.transform_audio:
            audio = self.transform_audio(audio)
        if self.transform_video:
            video = self.transform_video(video)

        # Get prompts, input_ids, and labels
        prompts = self.get_prompt(sample_1, sample_2)
        if self.max_input_length > 0:
            prompts["input_ids"] = prompts["input_ids"][:self.max_input_length]
            prompts["labels"] = prompts["labels"][:self.max_input_length]

        # Return audio, video, and some additional information if needed.
        return {
            'audio': audio,
            'video': video,
            "uid": sample_1.get("Uid", ""),
            "ground_truth": sample_1.get("Caption", ""),
            **prompts,
        }

    def get_prompt(self, sample_1, sample_2):
        assert self.tokenizer is not None
        new_dict = {}

        asr_nhyps = sample_1[self.nhyps_key_asr]["hyps"]
        vsr_nhyps = sample_2[self.nhyps_key_vsr]["hyps"]

        asr_best_hypothesis = asr_nhyps[0]
        vsr_best_hypothesis = vsr_nhyps[0]

        if self.max_nhyps is not None:
            asr_other_hypothesis = asr_nhyps[1:self.max_nhyps]
            vsr_other_hypothesis = vsr_nhyps[1:self.max_nhyps]
        else:
            asr_other_hypothesis = asr_nhyps[1:]
            vsr_other_hypothesis = vsr_nhyps[1:]

        if self.random_sample_nhyps:
            asr_other_hypothesis = random_sample_sequence(asr_other_hypothesis, len(asr_other_hypothesis))
            vsr_other_hypothesis = random_sample_sequence(vsr_other_hypothesis, len(vsr_other_hypothesis))

        final_prompt_no_response = self.prompt_1.replace("<<<ASR_NHYPS>>>", asr_best_hypothesis).replace("<<<VSR_NHYPS>>>", vsr_best_hypothesis) + \
            self.prompt_2.replace("<<<ASR_NHYPS>>>", '\n'.join(asr_other_hypothesis)).replace("<<<VSR_NHYPS>>>", '\n'.join(vsr_other_hypothesis)) + self.prompt_3
        final_prompt = final_prompt_no_response + sample_1["Caption"] + self.tokenizer.eos_token # </s>

        if self.apply_chat_template:
            # Use chat template for models like phi-3.5
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": final_prompt_no_response},
            ]
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # append <|assistant>| at the end
            )
            answer_ids = self.tokenizer(sample_1["Caption"], add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
            input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.int64)
            input_ids_no_response = torch.tensor(prompt_ids, dtype=torch.int64)
            labels = torch.tensor([-1] * len(prompt_ids) + answer_ids, dtype=torch.int64)
        else:
            # Use traditional tokenization
            input_ids_no_response = self.tokenizer.encode(final_prompt_no_response)
            input_ids = self.tokenizer.encode(final_prompt)
            labels_temp = input_ids.copy()
            labels = [-1] * len(input_ids_no_response) + labels_temp[len(input_ids_no_response):]
            
            input_ids_no_response = torch.tensor(input_ids_no_response, dtype=torch.int64)
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
            labels = torch.tensor(labels, dtype=torch.int64)

        new_dict['input_ids_no_response'] = input_ids_no_response
        new_dict['input_ids'] = input_ids
        new_dict['labels'] = labels
        new_dict['input'] = final_prompt

        return new_dict
        

class DualHypothesesMaskAVDataset(DualHypothesesAVDataset):
    def __init__(self, 
                 split, 
                 json_path,
                 prompts_format="RelPrompt",
                 leave_masks=False,
                 mask_threshold=None,
                 time_window=0.4,
                 **kwargs):
        super().__init__(split, json_path, prompts_format=prompts_format, **kwargs)
        self.leave_masks = leave_masks
        self.mask_threshold = mask_threshold
        self.audio_chunk_size = int(16000 * time_window)  # 16kHz audio sampling
        self.video_chunk_size = int(25 * time_window)  # 25fps video sampling

    def get_noise_mask(self, sample, modality="audio"):
        """
        Generate noise mask for audio or video.
        Args:
            sample (dict): Sample data containing corruption info.
            modality (str): Either "audio" or "video".
        Returns:
            List of frame labels ('C', 'N').
        """
        if modality == "audio":
            total_len = sample["Audio_Corruption"]["total_len"]
            occ_len = sample["Audio_Corruption"]["occ_len"]
            start_fr = sample["Audio_Corruption"]["start_fr"]
            snr = sample["Audio_Corruption"]["snr"]
        elif modality == "video":
            total_len = sample["Visual_Corruption"]["total_len"]
            occ_len = sample["Visual_Corruption"]["occ_len"]
            start_fr = sample["Visual_Corruption"]["start_fr"]
            snr = -100
        else:
            raise ValueError("Invalid modality. Choose 'audio' or 'video'.")

        mask = ['C'] * total_len
        if self.mask_threshold is None or snr < self.mask_threshold:
            mask[start_fr:start_fr + occ_len] = ['N'] * occ_len
        return mask

    def chunk_reliability_score(self, mask, modality="audio", chunk_size=10, prefix=""):
        """
        Compute reliability scores and bin labels for each chunk in the mask.
        Args:
            mask (list): List of frame labels (e.g., 'C', 'M', 'N').
            modality (str): Either "audio" or "video".
        Returns:
            Tuple of chunk scores (fraction of clean frames per chunk) and bin labels.
        """
        scores = []
        bin_labels = []
        
        for i in range(0, len(mask), chunk_size):
            chunk = mask[i:i + chunk_size]
            clean_frames = chunk.count('C')
            score = clean_frames / len(chunk)
            scores.append(score)
            
            # Assign bin label based on chunk reliability
            if score > 0.9:
                bin_labels.append(f"<<{prefix}C>>")
            elif score < 0.6:
                bin_labels.append(f"<<{prefix}N>>")
            else:
                bin_labels.append(f"<<{prefix}M>>")
        
        return scores, bin_labels

    def __getitem__(self, idx):
        # Get the JSON sample
        uid = self.idx2uid[idx]
        sample_1, sample_2 = random.choices(self.uid2sample[uid], k=2)

        audio = self.load_audio(sample_1)
        video = self.load_video(sample_2)
        video = self.lipreading_preprocessing_func(torch.from_numpy(video))

        # Apply optional transforms
        if self.transform_audio:
            audio = self.transform_audio(audio)
        if self.transform_video:
            video = self.transform_video(video)

        # Get noise mask, reliability scores, and bin labels
        if self.audio_corruption_enabled:
            audio_mask = self.get_noise_mask(sample_1, modality="audio")
        else:
            audio_mask = ['C'] * len(audio)
        if self.visual_corruption_enabled:
            video_mask = self.get_noise_mask(sample_2, modality="video")
        else:
            video_mask = ['C'] * len(video)
        audio_scores, audio_bin_labels = self.chunk_reliability_score(audio_mask, modality="audio", chunk_size=self.audio_chunk_size, prefix="")
        video_scores, video_bin_labels = self.chunk_reliability_score(video_mask, modality="video", chunk_size=self.video_chunk_size, prefix="")

        # Get prompts, input_ids, and labels
        prompts = self.get_prompt(sample_1, sample_2, audio_bin_labels, video_bin_labels)
        if self.max_input_length > 0:
            prompts["input_ids"] = prompts["input_ids"][:self.max_input_length]
            prompts["labels"] = prompts["labels"][:self.max_input_length]

        # Return audio, video, and additional information
        return {
            'audio': audio,
            'video': video,
            "uid": sample_1.get("Uid", ""),
            "ground_truth": sample_1.get("Caption", ""),
            "audio_bin_labels": audio_bin_labels,
            "video_bin_labels": video_bin_labels,
            **prompts,
        }

    def get_prompt(self, sample_1, sample_2, audio_bin_labels, video_bin_labels):
        assert self.tokenizer is not None
        new_dict = {}

        asr_nhyps = sample_1[self.nhyps_key_asr]["hyps"]
        vsr_nhyps = sample_2[self.nhyps_key_vsr]["hyps"]

        asr_best_hypothesis = asr_nhyps[0]
        vsr_best_hypothesis = vsr_nhyps[0]

        if self.max_nhyps is not None:
            asr_other_hypothesis = asr_nhyps[1:self.max_nhyps]
            vsr_other_hypothesis = vsr_nhyps[1:self.max_nhyps]
        else:
            asr_other_hypothesis = asr_nhyps[1:]
            vsr_other_hypothesis = vsr_nhyps[1:]

        if self.random_sample_nhyps:
            asr_other_hypothesis = random_sample_sequence(asr_other_hypothesis, len(asr_other_hypothesis))
            vsr_other_hypothesis = random_sample_sequence(vsr_other_hypothesis, len(vsr_other_hypothesis))

        final_prompt_no_response = self.prompt_1.replace("<<<ASR_BEST_NHYPS>>>", asr_best_hypothesis).replace("<<<VSR_BEST_NHYPS>>>", vsr_best_hypothesis).replace("<<<ASR_NHYPS>>>", '\n'.join(asr_other_hypothesis)).replace("<<<VSR_NHYPS>>>", '\n'.join(vsr_other_hypothesis))
        if not self.leave_masks:
            final_prompt_no_response = final_prompt_no_response.replace("<<<ASR_MASKS>>>", ''.join(audio_bin_labels)).replace("<<<VSR_MASKS>>>", ''.join(video_bin_labels))
        final_prompt_no_response += self.prompt_3
        final_prompt = final_prompt_no_response + sample_1["Caption"] + self.tokenizer.eos_token # </s>

        if self.apply_chat_template:
            # Use chat template for models like phi-3.5
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": final_prompt_no_response},
            ]
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # append <|assistant>| at the end
            )
            answer_ids = self.tokenizer(sample_1["Caption"], add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
            input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.int64)
            input_ids_no_response = torch.tensor(prompt_ids, dtype=torch.int64)
            labels = torch.tensor([-1] * len(prompt_ids) + answer_ids, dtype=torch.int64)
        else:
            # Use traditional tokenization
            input_ids_no_response = self.tokenizer.encode(final_prompt_no_response)
            input_ids = self.tokenizer.encode(final_prompt)
            labels_temp = input_ids.copy()
            labels = [-1] * len(input_ids_no_response) + labels_temp[len(input_ids_no_response):]
            
            input_ids_no_response = torch.tensor(input_ids_no_response, dtype=torch.int64)
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
            labels = torch.tensor(labels, dtype=torch.int64)

        new_dict['input_ids_no_response'] = input_ids_no_response
        new_dict['input_ids'] = input_ids
        new_dict['labels'] = labels
        new_dict['input_no_response'] = final_prompt_no_response
        new_dict['input'] = final_prompt

        return new_dict

    def collate_fn(self, samples):
        inputs = [item["input"] for item in samples]
        inputs_no_response = [item["input_no_response"] for item in samples]
        uids = [item["uid"] for item in samples]
        ground_truths = [item["ground_truth"] for item in samples]
        input_ids_list = [item["input_ids"] for item in samples]
        input_ids_no_response_list = [item["input_ids_no_response"] for item in samples]
        labels_list = [item["labels"] for item in samples]
        audio_list = [item["audio"] for item in samples]
        video_list = [item["video"] for item in samples]
        audio_bin_labels_list = [item["audio_bin_labels"] for item in samples]
        video_bin_labels_list = [item["video_bin_labels"] for item in samples]

        # Pad text sequences
        max_len = max(seq.size(0) for seq in input_ids_list)
        def pad_right(seq, pad_value):
            n = max_len - seq.size(0)
            return torch.cat([seq, torch.full((n,), pad_value, dtype=seq.dtype)])
        
        input_ids = torch.stack([pad_right(seq, 0) for seq in input_ids_list])
        labels = torch.stack([pad_right(seq, -1) for seq in labels_list])

        # Pad audio/video sequences using pad_sequence.
        audio = pad_sequence(audio_list, batch_first=True)
        video_list, _ = pad_mouth(video_list)
        video = [torch.FloatTensor(video).unsqueeze(0) for video in video_list]
        video = torch.stack([x for x in video], dim=0)

        return {
            "input": inputs,
            "input_no_response": inputs_no_response,
            "uid": uids,
            "ground_truth": ground_truths,
            "input_ids": input_ids,
            "input_ids_no_response": input_ids_no_response_list,
            "labels": labels,
            "audio": audio,
            "video": video,
            "audio_bin_labels": audio_bin_labels_list,
            "video_bin_labels": video_bin_labels_list,
        }
