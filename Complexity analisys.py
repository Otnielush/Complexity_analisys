import pandas as pd
import numpy as np
import sys, os
from algorithms import Word_entrophy, Relative_entropy, TTR, MATTR


sys.path.insert(0, sys.path[0]+'/cwi/CWI Sequence Labeller/sequence-labeler-master/')
from complex_labeller import Complexity_labeller

model_path = sys.path[0]+'/cwi_seq.model'
temp_path = './temp_file.txt'


def loader():
    global CL
    CL = Complexity_labeller(model_path, temp_path)
    print('CWI model loaded')
    print()


# input DataFrame pandas
def analyse(table_results):
    if table_results.shape[0] == 0:
        table_new = True
    else:
        table_new = False
    # open directory as kindergarten
    kindergarten_num = 0
    file_num_total = 0
    for kindergarten in os.listdir(path):
        path_kindergarten = os.path.join(path, kindergarten)
        # directory check
        if not os.path.isdir(path_kindergarten):
            continue

        kindergarten_num += 1
        record = {}
        record['Kindergarten'] = os.path.split(path_kindergarten)[-1]

        child_num = 0
        for child in os.listdir(path_kindergarten):
            child_num += 1
            path_child = os.path.join(path_kindergarten, child)
            # directory check
            if not os.path.isdir(path_child):
                continue
            # directory check
            for finish in os.listdir(path_child):
                if not finish == 'finished':
                    continue

                record['Child name'] = child
                path_finish = os.path.join(path_child, finish)
                file_num = 0

                # open files for analysis
                for file in os.listdir(path_finish):
                    path_file = os.path.join(path_finish, file)
                    # file check
                    if not os.path.isfile(path_file):
                        continue

                    file_num += 1
                    file_num_total += 1
                    print('\rKG: "{}" {:2.2f}%. file {} of {}. Total {:2.2f}%. file {} of {} | '.format(
                       record['Kindergarten'], file_num/num_files[kindergarten_num-1]*100, file_num, num_files[kindergarten_num-1],
                        file_num_total/num_files[-1]*100, file_num_total, num_files[-1]
                    ), end='')

                    record['File name'] = os.path.split(path_file)[-1].split('.')[0]
                    name = file.split('.')[0]
                    if len(name) < 6:
                        name = '0'+name
                    record['Age'] = int(name[:2]) + int(name[2:4])/12 + int(name[4:6])/365

                    # #condition for Dataset editing
                    # if table_results[
                    #     (table_results['Kindergarten'] == 'Inkelas') & (table_results['File name'] == record['File name'])]['cwi complexity'].values == -1000:
                    #     print('\nFile:', record['File name'])
                    # else:
                    #     continue

                    with open(path_file, 'r') as f:
                        text = f.readlines()
                    text = ''.join(text).replace('\n', ' ').replace('_', ' ')

                    try:
                        record['Word entropy'] = Word_entrophy(text)
                    except:
                        record['Word entropy'] = -1000
                    try:
                        record['Relative entropy of word structure'] = 0  # Relative_entropy(text)
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

                    text = text.replace('.', '')
                    try:
                        CL.convert_format_string(text)
                        print('1')
                        probs = CL.get_prob_labels()
                        record['cwi complexity'] = np.mean(probs)
                        print('1')
                        record['cwi std'] = np.std(probs)
                        print('1')
                        record['cwi min'] = np.min(probs)
                        print('1')
                        record['cwi max'] = np.max(probs)
                        print('1')
                        record['cwi probs'] = probs
                    except:
                        record['cwi complexity'] = -1000

                    if table_new:
                        table_results = table_results.append(record, ignore_index=True)
                    else:
                        table_results.loc[file_num_total-1, :] = record.values()


    print()
    table_results.to_csv('complexity_analysis_results.csv', index=None)
    table_results.to_excel('complexity_analysis_results.xlsx', index=None)

    return



def check_dirs():
    if len(sys.argv) > 1:
        path = os.path.join('.', sys.argv[1])
        print(path)
    else:
        path = '.\data\Kindergartens'

    num_files = []

    for kindergarten in os.listdir(path):
        path_kindergarten = os.path.join(path, kindergarten)
        # directory check
        if not os.path.isdir(path_kindergarten):
            continue

        num_files.append(0)
        child_num = 0
        for child in os.listdir(path_kindergarten):
            child_num += 1
            path_child = os.path.join(path_kindergarten, child)
            # directory check
            if not os.path.isdir(path_child):
                continue
            # directory check
            for finish in os.listdir(path_child):
                if not finish == 'finished':
                    continue

                num_files[-1] += len(os.listdir(os.path.join(path_child, finish)))

        print('{} - ({}) files/dirs'.format(kindergarten, num_files[-1]))
    num_files.append(sum(num_files))
    print('Total:', num_files[-1])
    print()
    return path, num_files



if __name__ == "__main__":

    path, num_files = check_dirs()
    table_results = pd.DataFrame([], columns=['Kindergarten', 'Child name', 'Age', 'File name', 'cwi complexity', 'cwi std', 'cwi min', 'cwi max',
                                              'Word entropy', 'Relative entropy of word structure', 'TTR', "MTTR", 'cwi probs'])
    loader()
    table_results = pd.read_csv('complexity_analysis_results.csv')
    analyse(table_results)
