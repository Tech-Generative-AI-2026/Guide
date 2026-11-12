"""암호문 문자열 <-> integer ID 변환 (문자 단위).

<SEP>와 <EOS>는 각각 하나의 ID로 처리한다. decode(encode(x)) == x 가 공개 데이터에서 성립해야 한다.
학생은 이 파일을 수정할 수 있다. 평가 시 evaluator는 encode, decode, SEP_ID, EOS_ID만 사용한다.
"""
import re

CHARS = "BDFHJKMQVWXZbdfhjkmqvwxz¿ɐɑɒɓɔɕɖɗɘəɛɜɞɟɠɡɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴαβγδεζηθικλμνξπρστυφχψω†↑∆∑∗⟨⟩"
_stoi = {c: i for i, c in enumerate(CHARS)}
_itos = {i: c for i, c in enumerate(CHARS)}
SEP_ID = len(CHARS)
EOS_ID = len(CHARS) + 1
_stoi["<SEP>"] = SEP_ID
_stoi["<EOS>"] = EOS_ID
_itos[SEP_ID] = "<SEP>"
_itos[EOS_ID] = "<EOS>"
VOCAB_SIZE = len(CHARS) + 2
_SPECIAL = re.compile(r"(<SEP>|<EOS>)")


def encode(text):
    ids = []
    for part in _SPECIAL.split(text):
        if part in ("<SEP>", "<EOS>"):
            ids.append(_stoi[part])
        else:
            ids.extend(_stoi[c] for c in part)
    return ids


def decode(ids):
    return "".join(_itos[i] for i in ids)
