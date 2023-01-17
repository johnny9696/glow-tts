#lang number
"""
독일어 0
영어 1
프랑스어 2
스페인어 3
한국어 
"""

def l2num(lang) -> int:
    result=None
    if lang == 'dutch' or lang == 'Dutch' or lang == 'Du' or lang == 'du':
        result = 0
    elif lang == 'english' or lang == 'English' or lang == 'En' or lang == 'en':
        result = 1
    elif lang == 'franch' or lang == 'Franch' or lang == 'Fr' or lang == 'fr':
        result=2
    elif lang == 'spanish' or lang == 'Spanish' or lang == 'Es' or lang == 'es':
        result=3
    return result