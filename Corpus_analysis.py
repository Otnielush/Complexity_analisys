import sys, os

import numpy as np
from algorithms import Word_entrophy, Relative_entropy, TTR, MATTR
import pandas as pd


def analyse():
    table_results = pd.DataFrame([], columns=['Corpus', 'File name', 'Word entropy', 'Relative entropy of word structure', 'TTR', "MTTR"])
    # open directory as corpus
    corpus_num = 0
    file_num_total = 0
    for corpus in os.listdir(path):
        corpus_num += 1
        record = {}
        path_corpus = os.path.join(path, corpus)
        record['Corpus'] = os.path.split(path_corpus)[-1]
        # open files for analysis
        file_num = 0


        for file in os.listdir(path_corpus):
            file_num += 1
            file_num_total += 1
            print('\rCorpus: "{}" {:2.2f}%. file {} of {}. Total {:2.2f}%. file {} of {} | '.format(
               record['Corpus'], file_num/num_files[corpus_num-1]*100, file_num, num_files[corpus_num-1],
                file_num_total/sum(num_files)*100, file_num_total, sum(num_files)
            ), end='')


            path_file = os.path.join(path_corpus, file)
            if os.path.isfile(path_file):
                record['File name'] = os.path.split(path_file)[-1].split('.')[0]
                with open(path_file, 'r') as f:
                    text = f.readlines()

                try:
                    record['Word entropy'] = Word_entrophy(text)
                except:
                    record['Word entropy'] = -1000
                try:
                    record['Relative entropy of word structure'] = Relative_entropy(text)
                except:
                    record['Relative entropy of word structure'] = -1000
                try:
                    record['TTR'] = TTR(text)
                except:
                    record['TTR'] = -1000
                try:
                    record['MTTR'] = MATTR(text)
                except:
                    record['MTTR'] = -1000

                table_results = table_results.append(record, ignore_index=True)

    print()
    table_results.to_csv('corpus_analysis_results.csv')
    table_results.to_excel('corpus_analysis_results.xlsx')

    return


def check_dirs():
    if len(sys.argv) > 1:
        path = os.path.join('.', sys.argv[1])
        print(path)
    else:
        path = '.\data\MC-data'

    dir1 = os.listdir(path)
    num_files = []
    for dir in dir1:
        num_files.append(len(os.listdir(os.path.join(path, dir))))
        print('{} - ({}) files/dirs'.format(dir, num_files[-1]))
    print()
    return path, num_files




if __name__ == "__main__":
    path, num_files = check_dirs()

    print('First level of directories will be a Corpus')
    print('Analyse all this files?  y / any key')
    do_analyse = input()
    if do_analyse.lower() == 'y':
        analyse()
