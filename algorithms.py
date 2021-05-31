import numpy as np
import random
import re
import copy
import pandas as pd
from nltk.tokenize import word_tokenize
import itertools

import time
                                            # Word entrophy
# чем больше спектр типов слов(спряжений и тп) тем больше информации упаковывается в слово нежели в словосочетание или предложение.
# T - text
# V - vocabulary of word types {w1,w2,w3,w4,..,wVV} VV = |V|
# p(w) = Pr(T=w) for w in V - probability of word type

# The average information content of word types:
# H(T) = - sum( p(w_i) * log2(p(w_i)) ) from i=1 to VV
# jse - James-Stein shrinkage estimator
def Word_entrophy(text, jse=False, alpha=0.7):
    if type(text) == list:
        text = ' '.join(text)
    text = text.replace('\n', ' ').replace('  ', ' ').split(' ')

    # probability of word type
    # prob_dict = {x[0]:x[1]/len(text)   for x in np.transpose(np.unique(text, return_counts=True))}
    prob_dict = {}
    for w in text:
        if w in prob_dict:
            prob_dict[w] += 1
        else:
            prob_dict[w] = 1
    prob_dict = {x: prob_dict[x] / len(text) for x in prob_dict}

    # James-Stein shrinkage estimator
    if jse:
        shrinkage_prob = 1 / len(prob_dict)
        # max_prob = prob_dict
        # max_prob = prob_dict[max(prob_dict, key=prob_dict.get)]  # maximum probability from our text
        alpha_func = lambda x: round(alpha * shrinkage_prob + (1 - alpha) * x, 6)
        prob_dict = {x: alpha_func(prob_dict[x]) for x in prob_dict}

    return round(-sum(prob_dict[word] * np.log2((prob_dict[word])) for word in prob_dict), 5)




                                    # Relative entropy of word structure
# T - text drawn from alphabet A
# A = {c1,c2,c3,..cAA} - alphabet
#
# ~H(T) = [ sum( l_i / log2(i+1) ) / n ]^-1 for i {1:n}
#
# n - number of characters in text (T)
# l_i - is the length of the longest substring from position i onward that has not appeared before
#
# ~D = ~H(T_masked) - ~H(T_orig)
#
# T_masked - token of the same length but with characters randomly drawn with equal probability from the alphabet A
def longest_substring(text):
    # last index of every character
    last_idx = {}
    max_len = 0

    # starting index of current
    # window to calculate max_len
    start_idx = 0

    for i in range(0, len(string)):

        # Find the last index of str[i]
        # Update start_idx (starting index of current window)
        # as maximum of current value of start_idx and last
        # index plus 1
        if string[i] in last_idx:
            start_idx = max(start_idx, last_idx[string[i]] + 1)

        # Update result if we get a larger window
        max_len = max(max_len, i - start_idx + 1)

        # Update last index of current char.
        last_idx[string[i]] = i

    return max_len




def max_shortest_substring_d(text, i):
    max_len = 0
    start = 0
    exist = 1
    lenght = 0
    text_len = len(text)
    while True:
        while exist and (i + start + lenght) <= text_len:
            lenght += 1
            exist_old = exist
            exist = re.search(re.escape(text[i + start: i + start + lenght]), text[:i])
        if max_len < lenght and exist is None:
            max_len = lenght
        lenght = 0
        start += 1
        exist = 1
        if text_len - (i + start) < max_len:
            return max_len



def probability_char(text):
    text = ''.join(text).translate({ord(i): None for i in '“”<>,][.;:!?)/−(" ' + "'"})
    char_counts = {}
    char_prob = {}
    for i in range(len(text)):
        if text[i] in char_counts:
            char_counts[text[i]] += 1
        else:
            char_counts[text[i]] = 1
    char_prob = {x: char_counts[x] / len(text) for x in char_counts}
    return char_prob, char_counts


def calc_probability_word(word, dict_prob):
    summ = 0
    for w in word:
        summ += dict_prob[w]
    return summ


def make_random_copy_text(text):
    prob_char, _ = probability_char(text)
    text = text.translate({ord(i): None for i in '“”<>,][.;:!?)/−("' + "'"}).split(' ')
    text_c = copy.deepcopy(text)

    chars = [x for x in prob_char]

    prob_dev = 0.1
    # print('Making random copy of text', end='')
    for i in range(len(text)):
        # print('\rMaking random copy of text: {:2.2f}% '.format((i + 1) / len(text) * 100), end='')
        word_prob = calc_probability_word(text[i], prob_char)
        new_word_prob = 0

        while new_word_prob < word_prob * (1 - prob_dev) or new_word_prob > word_prob * (1 + prob_dev):
            new_word = ''
            new_word_prob = 0
            for l in range(len(text[i])):
                new_char = random.choice(chars)
                new_word += new_char
                new_word_prob += prob_char[new_char]

        text_c[i] = new_word
    # print('\rText copied with the same probability')
    return text_c


def make_random_copy_text2(text):
    text = text.translate({ord(i): None for i in '“”<>,][.;:!?)/−("' + "'"}).split(' ')
    text_c = copy.deepcopy(text)

    A = [chr(x) for x in np.arange(97, 97 + 26, 1)]  # Alphabet

    for i in range(len(text_c)):
        word_len = len(text_c[i])
        if word_len > 1:
            new_word = ''.join([random.choice(A) for _ in range(word_len)])
            text_c[i] = new_word
    return text_c


def isSub(t1, t2):
    lt2 = len(t2)

    startI = t1.find(t2)
    endI = startI + lt2

    if startI >= 0 and endI < len(t1):
        return lt2 + 1
    else:
        return 0


def max_shortest_substring3(text, i):
    part1, part2 = text[:i], text[i:]

    # Get all substrings of string
    # Using itertools.combinations()
    res = sorted(
        set([part1[x:y] for x, y in itertools.combinations(range(len(part1) + 1), r=2) if (y - x) <= len(part2)]),
        key=lambda x: len(x), reverse=True)
    for x in res:
        r = isSub(part2, x)
        if r > 0:
            # print(x)
            return r
    return 0

def max_shortest_substring(text, i):
    max_len = 0
    start = 0
    exist = 1
    lenght = 0
    text_len = len(text)
    while True:
        while exist and (i + start + lenght) <= text_len:
            lenght += 1
            exist_old = exist
            exist = re.search(re.escape(text[i + start: i + start + lenght]), text[:i])
        if max_len < lenght and exist is None:
            max_len = lenght
        lenght = 0
        start += 1
        exist = 1
        if text_len - (i + start) < max_len:
            return max_len

# input string of text
def Relative_entropy_old(text, redundancy=False):
    summ = 0
    if type(text) == list:
        text = ' '.join(text)

    text = text.replace('\n', ' ')
    # special = '“”<>,][.;:!?)/−*(" ' + "'"
    # text_2 = cleanText( text)
    text_2 = ' '.join(text.split(' ')).translate({ord(i): None for i in '“”<>,][.;:!?)/−*("' + "'"})
    text_2 = text_2.replace('  ', ' ')

    print('RE: 00.0%', end='')
    len_print = 9
    for i in np.arange(1, len(text_2), 1):    # <-------------- CHANGES HERE. starting from 1 because need to divide text to 2 parts
        # max length of the shortest substring from position i onward that has not appeared before
        l = max_shortest_substring(text_2, i)
        summ += (l / np.log2(i + 1))
        msg = 'RE: {:2.1f}%'.format((i + 1) / len(text_2) * 100)
        print(len_print*'\b'+msg, end='')
        len_print = len(msg)

    summ /= len(text_2)
    RE = pow(summ, -1)
    print('\b'*len_print, end='')

    # To estimate the amount of redundancy/predictability contributed by within-word structure, Koplenig
    # et al. (2016) replace each word token in T by a token of the same length but with characters randomly
    # drawn with equal probability from the alphabet A. The entropy of the original text is then subtracted
    # from the masked text
    if redundancy:
        RE_masked = Relative_entropy_old(getRandEntry(text), False)

        # The bigger D̂, the more information is stored within words, i.e. in morphological regularities. This
        # measure of morphological complexity is denoted CD in the following
        D = RE_masked - RE

        RE = D
        # print('Redundancy of RE:', round(RE, 6))

    return RE


    # ----------------------------------------------------------------------------------------
# TODO dict for copy text and dinamic max shortest....
def getRandEntry(text, lang='en'):
    if re.search('áéíó', text):
        lang = 'fr'
    if lang == 'en':
        A = [x for x in 'abcdefghigklmnopqrstuvwxyz']
    else:
        A = [x for x in 'abcdefghigklmnopqrstuvwxyzáéíóúüñèèçàêô']

    Dict_random = dict()
    for word in text.split(' '):
        if word in Dict_random.keys():
            continue
        Dict_random[word] = randomizeWord(len(word))

    res = []
    for x in text.split(' '):
        if len(x) > 1:
            res.append(Dict_random[x])
        else:
            res.append(x)

    text_n = " ".join(res)
    # print('text', text)
    # print('copy', text_n)
    return text_n


def cleanText(text):
    text = text.replace('\n', ' ')
    special = '“”<>,\]\[\.;:!\?\)/−\*\(" \''

    return re.sub(special, '', text)

def randomizeWord(l):
    A = [x for x in'abcdefghigklmnopqrstuvwxyz']
    return "".join([random.choice(A) for x in range(l)] )


def make_random_text(text):

    text_c =cleanText(text.lower())
    # text_c = copy.deepcopy(text)
    # A = [chr(x) for x in np.arange(97, 97 + 26, 1)]  # Alphabet
    res= []
    for x in text_c.split():
        if len(x)>1 :
            res.append(randomizeWord(len(x)))
        else:
            res.append(x)
    return " ".join(res)

def H(text):
    summ = 0
    # if type(text) == list:
    #     text = ' '.join(text)
    # text = text.replace('\n', ' ')
    #
    # text_2 = cleanText( text)
    # text_2 = ' '.join(text.split(' ')).translate({ord(i): None for i in '“”<>,][.;:!?)/−*("  ' + "'"})

    print('RE: 00.0%', end='')
    len_print = 9
    for i in np.arange(1, len(text), 1):    # <-------------- CHANGES HERE. starting from 1 because need to divide text to 2 parts
        # max length of the shortest substring from position i onward that has not appeared before
        l = max_shortest_substring(text, i)
        summ += (l / np.log2(i + 1))
        msg = 'RE: {:2.1f}%'.format((i + 1) / len(text) * 100)
        print(len_print * '\b' + msg, end='')
        len_print = len(msg)

    summ /= len(text)
    RE = pow(summ, -1)
    print('\b' * len_print, end='')
    return RE

def Relative_entropy(text):
    if type(text) == list:
        text = ' '.join(text)
    text = text.replace('\n', ' ')
    text_2 = cleanText(text)
    text_2 = text_2.replace('  ', ' ')
    # D = ~H(T_masked) - ~H(T_orig)
    return H(getRandEntry(text_2)) - H(text_2)


                                        # TTR - Type/Token ratios
# C_TTR = V / ( summ(fr_i) )
#
# V - number of all types of words(tokens)
# fr - frequency of i type
# input string of text
def TTR(text):
    if type(text) == list:
        text = ' '.join(text)
    text = text.translate({ord(i): None for i in '“”<>,][.;:!?)/−("'+"'"}).split(' ')
    pps = np.unique(text, return_counts=True)
    return len(pps[0])/sum(pps[1])


                                        # MATTR: TTR + window
# input string of text
def MATTR(text, window=500):
    if type(text) == list:
        text = ' '.join(text)

    if len(text) < window:  # <--------- CHANGES HERE
        window = len(text)
    pos = 0
    result = 0
    while pos + window <= len(text):
        result += TTR(text[pos: pos+window])
        pos += 1
    return result/pos




if __name__ == '__main__':

    tt = '''que una vez estaba mirando en el bote
estaban durmiendo que que que 
y se despertaron
y se le escapó la rana
luego el perro metió la cabeza en el frasco
luego se cayó por la ventana
y luego se fueron y
luego se fueron a un avispero
luego se metieron en el árbol en el
se cayó
y luego todas las avispas le fueron a picar al perro
está ya y éste
le coge un ciervo el perro
y luego se quedan
le tiran al niño a un lago
cogen a su rana otra vez
luego se van a casa
'''

    # pr = Word_entrophy(tt)
    # print(pr)
    start_time = time.time()
    re1 = Relative_entropy(copy.deepcopy(tt))
    print(re1, 'time', time.time()-start_time)

    start_time = time.time()
    re2 = Relative_entropy_old(copy.deepcopy(tt), True)
    print(re2, 'time', time.time() - start_time)

    ttr = TTR(tt)
    print(f"{ttr = }")
    mttr = MATTR(tt)
    print(f"{mttr = }")


