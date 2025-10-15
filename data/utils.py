#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/LICENSE

# Ack: Code taken from Pingchuan Ma: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks

import cv2
import h5py
import random
import numpy as np
import logging

import torch

__all__ = ['Compose', 'Normalize', 'CenterCrop', 'RgbToGray', 'RandomCrop',
           'HorizontalFlip', 'AddNoise', 'NormalizeUtterance',
           'get_preprocessing_pipelines', 'load_mouthroi', 'pad_mouth',
           'random_sample_sequence', 'word_emb_diff', 'sent_emb_diff', 'setup_logging']

class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RgbToGray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


class NormalizeUtterance():
    """Normalize per raw audio by removing the mean and divided by the standard deviation
    """
    def __call__(self, signal):
        signal_std = 0. if np.std(signal)==0. else np.std(signal)
        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std


class AddNoise(object):
    """Add SNR noise [-1, 1]
    """

    def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [np.float32, np.float64], "noise only supports float data type"
        
        self.noise = noise
        self.snr_levels = snr_levels

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2 **2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [np.float32, np.float64], "signal only supports float32 data type"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise)-len(signal))
            noise_clip = self.noise[start_idx:start_idx+len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*np.sqrt(factor)).astype(np.float32)
            return desired_signal
        

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
    if not filename:
        return np.zeros((1, 88, 88), dtype=np.float32)  # dummy
    if filename.endswith('.hdf5'):
        with h5py.File(filename, 'r') as hf:
            return hf['video_frames'][:]
    elif filename.endswith('.mp4'):
        cap = cv2.VideoCapture(filename)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
def pad_mouth(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        # print(sample.shape)
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )

    return collated_batch, lengths

def random_sample_sequence(lst, sample_size):
    indices = list(range(len(lst)))
    selected_indices = random.sample(indices, sample_size)
    selected_indices.sort()
    sampled_list = [lst[i] for i in selected_indices]
    return sampled_list

def word_emb_diff(reference, hypothesis, sbert_model):
    from data.my_jiwer.jiwer.measures import wer_embdiff
    output, edit_ops = wer_embdiff(reference, hypothesis)
    ref_words, hypo_words = output.references[0], output.hypotheses[0]

    emb_diffs = []
    for op in edit_ops:
        if op.tag == 'replace':
            ref_word, hypo_word = ref_words[op.src_pos], hypo_words[op.dest_pos]
        elif op.tag == 'delete':
            ref_word, hypo_word = ref_words[op.src_pos], None
        elif op.tag == 'insert':
            ref_word, hypo_word = None, hypo_words[op.dest_pos]
        else:
            continue

        ref_emb = torch.from_numpy(sbert_model.encode([ref_word], show_progress_bar=False)[0]) if ref_word else torch.zeros([384])
        hypo_emb = torch.from_numpy(sbert_model.encode([hypo_word], show_progress_bar=False)[0]) if hypo_word else torch.zeros([384])

        emb_diff = ref_emb - hypo_emb
        emb_diffs.append(emb_diff)

        # print('word', hypo_emb.mean(), ref_emb.mean(), emb_diff.mean())

    if len(emb_diffs) == 0:
        return torch.zeros([384])
    else:
        return torch.stack(emb_diffs, dim=0).mean(dim=0)

def sent_emb_diff(reference, hypothesis, sbert_model):
    embeddings = sbert_model.encode([reference, hypothesis], show_progress_bar=False)
    ref_emb, hypo_emb = torch.from_numpy(embeddings[0]), torch.from_numpy(embeddings[1])
    emb_diff = ref_emb - hypo_emb
    # print('sentence', hypo_emb.mean(), ref_emb.mean(), emb_diff.mean())

    return emb_diff

def setup_logging(log_file=None):
	"""
	Setup logging configuration for transcription errors.
	
	Args:
		log_file (str, optional): Path to log file. If None, logs to console only.
	"""
	logger = logging.getLogger('transcription_errors')
	logger.setLevel(logging.INFO)
	
	# Clear existing handlers
	for handler in logger.handlers[:]:
		logger.removeHandler(handler)
	
	# Create formatter
	formatter = logging.Formatter(
		'%(asctime)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	
	# Console handler
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)
	
	# File handler if specified
	if log_file:
		file_handler = logging.FileHandler(log_file, encoding='utf-8')
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	
	return logger
