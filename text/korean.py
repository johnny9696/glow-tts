#한국어 기반 TTS를 만들기 위해서 한국어 초성 중성 종성 기반한 셋을 구성함
first_letter='ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
middle_letter='ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
last_letter='ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'

symbols=[_pad]+list(_special)+list(_punctuation)+list(first_letter)+list(middle_letter)
last_letter=list(last_letter)
