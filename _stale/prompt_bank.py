PROMPT_TEMPLATES = [
    "a sound of {label} in the room",
    "the audio of {label}",
    "an indoor sound that resembles {label}",
]


CLASS_SYNONYMS = {
    "heavy_impact": ["heavy impact", "strong thud", "impact on floor"],
    "dragging": ["dragging", "scraping drag", "object dragging on floor"],
    "construction": ["construction", "construction work", "renovation noise"],
    "machine_noise": ["machine noise", "mechanical humming", "appliance machine sound"],
    "media_talking": ["media talking", "tv speech", "speaker talking audio"],
    "water_toilet": ["toilet water", "toilet flush", "bathroom flush sound"],
    "water_shower": ["shower water", "shower running", "bathroom shower sound"],
    "dog_bark": ["dog bark", "barking dog", "canine bark"],
    "others": ["other sound", "background noise", "non target sound"],
    "dummy_label": ["other sound", "background noise", "non target sound"],
}


def get_prompt_texts_for_class(class_name):
    synonyms = CLASS_SYNONYMS.get(class_name, [class_name.replace("_", " ")])
    prompts = []
    for synonym in synonyms:
        for template in PROMPT_TEMPLATES:
            prompts.append(template.format(label=synonym))
    return prompts
