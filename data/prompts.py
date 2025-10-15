# LLM prompts

GER_PROMPTS = {
    "prompt_1": 'Below is the best-hypotheses transcribed from speech recognition system. Please try to revise it using the words which are only included into other-hypothesis, and write the response for the true transcription.\n\n### Best-hypothesis:\n',
    "prompt_2": '\n\n### Other-hypothesis:',
    "prompt_3": '\n\n### Response:\n',
}

DualHyp_PROMPTS = {
    "prompt_1": 'Below are the best-hypothesis transcribed from speech recognition systems, ASR and VSR, respectively. Please try to revise it using the words which are only included into other-hypotheses, and write the response for the true transcription.\n\n### ASR Best-hypothesis:\n<<<ASR_NHYPS>>>\n\n### VSR Best-hypothesis:\n<<<VSR_NHYPS>>>',
    "prompt_2": '\n\n### ASR Other-hypotheses:\n<<<ASR_NHYPS>>>\n\n### VSR Other-hypotheses:\n<<<VSR_NHYPS>>>',
    "prompt_3": '\n\n### Response:\n',
}

RelPrompt_PROMPTS = {
    "prompt_1": 'Below are the best-hypothesis transcribed from speech recognition systems, ASR and VSR, respectively. Please try to revise it using the words which are only included into other-hypotheses, and write the response for the true transcription. Refer to the audio and video masks for reliability.\n\n\n### ASR Best-hypothesis:\n<<<ASR_BEST_NHYPS>>>\n\n### ASR Other-hypotheses:\n<<<ASR_NHYPS>>>\n\n### Audio Mask:\n<<<ASR_MASKS>>>\n\n\n### VSR Best-hypothesis:\n<<<VSR_BEST_NHYPS>>>\n\n### VSR Other-hypotheses:\n<<<VSR_NHYPS>>>\n\n### Video Mask:\n<<<VSR_MASKS>>>',
    "prompt_2": '',
    "prompt_3": '\n\n\n### Response:\n',
}


def get_prompts_format(name):
    """
    Return the prompts for the given name.
    """
    if name == "GER":
        return GER_PROMPTS
    elif name == "DualHyp":
        return DualHyp_PROMPTS
    elif name == "RelPrompt":
        return RelPrompt_PROMPTS
    else:
        raise ValueError(f"Unknown prompt name: {name}")