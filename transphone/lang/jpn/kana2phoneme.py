from .conv_table import FULL_KANA
from phonepiece.inventory import read_inventory
from transphone.config import TransphoneConfig

_kana2phonemes = {
    'ア': 'a',
    'イ': 'i',
    'ウ': 'u',
    'エ': 'e',
    'オ': 'o',
    'カ': 'k a',
    'キ': 'k i',
    'ク': 'k u',
    'ケ': 'k e',
    'コ': 'k o',
    'ガ': 'g a',
    'ギ': 'g i',
    'グ': 'g u',
    'ゲ': 'g e',
    'ゴ': 'g o',
    'サ': 's a',
    'シ': 'sh i',
    'ス': 's u',
    'セ': 's e',
    'ソ': 's o',
    'ザ': 'z a',
    'ジ': 'j i',
    'ズ': 'z u',
    'ゼ': 'z e',
    'ゾ': 'z o',
    'タ': 't a',
    'チ': 'ch i',
    'ツ': 't͡s u',
    'テ': 't e',
    'ト': 't o',
    'ダ': 'd a',
    'ヂ': 'j i',
    'ヅ': 'z u',
    'デ': 'd e',
    'ド': 'd o',
    'ナ': 'n a',
    'ニ': 'n i',
    'ヌ': 'n u',
    'ネ': 'n e',
    'ノ': 'n o',
    'ハ': 'h a',
    'ヒ': 'h i',
    'フ': 'f u',
    'ヘ': 'h e',
    'ホ': 'h o',
    'バ': 'b a',
    'ビ': 'b i',
    'ブ': 'b u',
    'ベ': 'b e',
    'ボ': 'b o',
    'パ': 'p a',
    'ピ': 'p i',
    'プ': 'p u',
    'ペ': 'p e',
    'ポ': 'p o',
    'マ': 'm a',
    'ミ': 'm i',
    'ム': 'm u',
    'メ': 'm e',
    'モ': 'm o',
    'ラ': 'r a',
    'リ': 'r i',
    'ル': 'r u',
    'レ': 'r e',
    'ロ': 'r o',
    'ワ': 'w a',
    'ヲ': 'o',
    'ヤ': 'y a',
    'ユ': 'y u',
    'ヨ': 'y o',
    'キャ': 'ky a',
    'キュ': 'ky u',
    'キョ': 'ky o',
    'ギャ': 'gy a',
    'ギュ': 'gy u',
    'ギョ': 'gy o',
    'シャ': 'sh a',
    'シュ': 'sh u',
    'ショ': 'sh o',
    'ジャ': 'j a',
    'ジュ': 'j u',
    'ジョ': 'j o',
    'チャ': 'ch a',
    'チュ': 'ch u',
    'チョ': 'ch o',
    'ニャ': 'ny a',
    'ニュ': 'ny u',
    'ニョ': 'ny o',
    'ヒャ': 'hy a',
    'ヒュ': 'hy u',
    'ヒョ': 'hy o',
    'ビャ': 'by a',
    'ビュ': 'by u',
    'ビョ': 'by o',
    'ピャ': 'py a',
    'ピュ': 'py u',
    'ピョ': 'py o',
    'ミャ': 'my a',
    'ミュ': 'my u',
    'ミョ': 'my o',
    'リャ': 'ry a',
    'リュ': 'ry u',
    'リョ': 'ry o',
    'イェ': 'i e',
    'シェ': 'sh e',
    'ジェ': 'j e',
    'ティ': 't i',
    'トゥ': 't u',
    'チェ': 'ch e',
    'ツァ': 't͡s a',
    'ツィ': 't͡s i',
    'ツェ': 't͡s e',
    'ツォ': 't͡s o',
    'ディ': 'd i',
    'ドゥ': 'd u',
    'デュ': 'd u',
    'ニェ': 'n i e',
    'ヒェ': 'h e',
    'ファ': 'f a',
    'フィ': 'f i',
    'フェ': 'f e',
    'フォ': 'f o',
    'フュ': 'hy u',
    'ブィ': 'b i',
    'ミェ': 'm e',
    'ウィ': 'w i',
    'ウェ': 'w e',
    'ウォ': 'w o',
    'クヮ': 'k a',
    'グヮ': 'g a',
    'スィ': 's u i',
    'ズィ': 'j i',
    'テュ': 't e y u',
    'ヴァ': 'b a',
    'ヴィ': 'b i',
    'ヴ': 'b u',
    'ヴェ': 'b e',
    'ヴォ': 'b o',
    'ン': 'N',
    'ッ': 'q',
    'ー': 'ː'
}

import re

class Kana2Phoneme:
    def __init__(self):

        self.phoneme_set = set(read_inventory('jpn').phoneme.elems)

        self._dict1 = {
            'キャ': 'ky a ',
            'キュ': 'ky u ',
            'キョ': 'ky o ',
            'ギャ': 'gy a ',
            'ギュ': 'gy u ',
            'ギョ': 'gy o ',
            'シャ': 'sh a ',
            'シュ': 'sh u ',
            'ショ': 'sh o ',
            'ジャ': 'j a ',
            'ジュ': 'j u ',
            'ジョ': 'j o ',
            'チャ': 'ch a ',
            'チュ': 'ch u ',
            'チョ': 'ch o ',
            'ニャ': 'ny a ',
            'ニュ': 'ny u ',
            'ニョ': 'ny o ',
            'ヒャ': 'hy a ',
            'ヒュ': 'hy u ',
            'ヒョ': 'hy o ',
            'ビャ': 'by a ',
            'ビュ': 'by u ',
            'ビョ': 'by o ',
            'ピャ': 'py a ',
            'ピュ': 'py u ',
            'ピョ': 'py o ',
            'ミャ': 'my a ',
            'ミュ': 'my u ',
            'ミョ': 'my o ',
            'リャ': 'ry a ',
            'リュ': 'ry u ',
            'リョ': 'ry o ',
            'イェ': 'i e ',
            'シェ': 'sh e ',
            'ジェ': 'j e ',
            'ティ': 't i ',
            'トゥ': 't u ',
            'チェ': 'ch e ',
            'ツァ': 't͡s a ',
            'ツィ': 't͡s i ',
            'ツェ': 't͡s e ',
            'ツォ': 't͡s o ',
            'ディ': 'd i ',
            'ドゥ': 'd u ',
            'デュ': 'd u ',
            'ニェ': 'n i e ',
            'ヒェ': 'h e ',
            'ファ': 'f a ',
            'フィ': 'f i ',
            'フェ': 'f e ',
            'フォ': 'f o ',
            'フュ': 'hy u ',
            'ブィ': 'b i ',
            'ミェ': 'm e ',
            'ウィ': 'w i ',
            'ウェ': 'w e ',
            'ウォ': 'w o ',
            'クヮ': 'k a ',
            'グヮ': 'g a ',
            'スィ': 's u i ',
            'ズィ': 'j i ',
            'テュ': 't e y u ',
            'ヴァ': 'b a ',
            'ヴィ': 'b i ',
            'ヴ': 'b u ',
            'ヴェ': 'b e ',
            'ヴォ': 'b o ',
        }
        self._dict2 = {
            'ア': 'a ',
            'イ': 'i ',
            'ウ': 'u ',
            'エ': 'e ',
            'オ': 'o ',
            'カ': 'k a ',
            'キ': 'k i ',
            'ク': 'k u ',
            'ケ': 'k e ',
            'コ': 'k o ',
            'ガ': 'g a ',
            'ギ': 'g i ',
            'グ': 'g u ',
            'ゲ': 'g e ',
            'ゴ': 'g o ',
            'サ': 's a ',
            'シ': 'sh i ',
            'ス': 's u ',
            'セ': 's e ',
            'ソ': 's o ',
            'ザ': 'z a ',
            'ジ': 'j i ',
            'ズ': 'z u ',
            'ゼ': 'z e ',
            'ゾ': 'z o ',
            'タ': 't a ',
            'チ': 'ch i ',
            'ツ': 't͡s u ',
            'テ': 't e ',
            'ト': 't o ',
            'ダ': 'd a ',
            'ヂ': 'j i ',
            'ヅ': 'z u ',
            'デ': 'd e ',
            'ド': 'd o ',
            'ナ': 'n a ',
            'ニ': 'n i ',
            'ヌ': 'n u ',
            'ネ': 'n e ',
            'ノ': 'n o ',
            'ハ': 'h a ',
            'ヒ': 'h i ',
            'フ': 'f u ',
            'ヘ': 'h e ',
            'ホ': 'h o ',
            'バ': 'b a ',
            'ビ': 'b i ',
            'ブ': 'b u ',
            'ベ': 'b e ',
            'ボ': 'b o ',
            'パ': 'p a ',
            'ピ': 'p i ',
            'プ': 'p u ',
            'ペ': 'p e ',
            'ポ': 'p o ',
            'マ': 'm a ',
            'ミ': 'm i ',
            'ム': 'm u ',
            'メ': 'm e ',
            'モ': 'm o ',
            'ラ': 'r a ',
            'リ': 'r i ',
            'ル': 'r u ',
            'レ': 'r e ',
            'ロ': 'r o ',
            'ワ': 'w a ',
            'ヲ': 'o ',
            'ヤ': 'y a ',
            'ユ': 'y u ',
            'ヨ': 'y o ',
            'ン': 'N ',
            'ッ': 'q ',
            'ー': 'ː ',
            'ァ': 'a ',
            'ィ': 'i ',
            'ゥ': 'u ',
            'ェ': 'e ',
            'ォ': 'o ',
            'ャ': 'y a ',
            'ュ': 'y u ',
            'ョ': 'y o ',
            'ヮ': 'w a ',
            'ヵ': 'k a ',
            'ヶ': 'k e ',
            'ヰ': 'i ',
            'ヱ': 'e ',
            'ヴ': 'b u ',
            'ヽ': '',  # this should not reach here though
            'ヾ': '',  # this should not reach here though
            'ヷ': 'w a ',
            'ヸ': 'i ',
            'ヹ': 'e ',
            'ヺ': 'o ',
            'ヿ': '',  # this should not reach here though
            '゛': '',  # this should not reach here though
            '゜': '',  # this should not reach here though
            'ヽ': '',  # this should not reach here though
            'ヾ': '',  # this should not reach here though
            'ゝ': '',  # this should not reach here though
            'ゞ': '',  # this should not reach here though
            '〆': '',  # this should not reach here though
            '々': '',  # this should not reach here though
        }
        self._regex1 = re.compile(u"(%s)" % u"|".join(map(re.escape, self._dict1.keys())))
        self._regex2 = re.compile(u"(%s)" % u"|".join(map(re.escape, self._dict2.keys())))

    def validate(self, text):
        for word in text.strip():
            if word not in FULL_KANA:
                return False
        return True

    def convert(self, origin_text):

        if isinstance(origin_text, list):
            text = ' '.join(origin_text)
        else:
            text = origin_text

        ret = text
        ret = self._regex1.sub(lambda m: self._dict1[m.string[m.start():m.end()]], ret)
        ret = self._regex2.sub(lambda m: self._dict2[m.string[m.start():m.end()]], ret)

        temp_phonemes = ret.split()

        phonemes = []
        for temp_phoneme in temp_phonemes:
            if temp_phoneme == 'ː':
                if len(phonemes) > 0 and phonemes[-1] in ['a', 'i', 'u', 'e', 'o']:
                    phonemes[-1] = phonemes[-1]+'ː'
                continue

            if temp_phoneme in self.phoneme_set:
                phonemes.append(temp_phoneme)
            else:
                TransphoneConfig.logger.error("Unknown phoneme: %s" % temp_phoneme)

        return phonemes
