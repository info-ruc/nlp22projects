import json
import json
import tarfile
from tqdm import tqdm
import argparse
import re
import pickle
from constant import *

INDEX_FILE_NAME = '{}_index.json'
OUTPUT_FILE_NAME = '{}_02.json'
STOCK_PATTERN = re.compile(r'\d+')
IMG_PATTERN = re.compile(r'<im.+?title="(.+?)"/>')
SUB_PATTERN = re.compile(r'\<.*\>')


def save_data(data, year): # 保存数据
    data_file = './data/matched_{}.pkl'.format(year)
    pickle.dump(data, open(data_file, 'wb'))

def read_data(year): # 读取数据
    data_file = './data/matched_{}.pkl'.format(year)
    data = pickle.load(open(data_file, 'rb'))
    return data

def read_file(filepath): 
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def read_and_match_tar_file(index_file, output_file, year):
    index_tar = tarfile.open(index_file,encoding='utf-8') # 读取index tar file
    output_tar = tarfile.open(output_file,encoding='utf-8') # output tar file

    index_dict = {STOCK_PATTERN.search(x.name).group():x for x in tqdm(index_tar.getmembers(), 'read index members') if '.json' in x.name}
    output_dict = {STOCK_PATTERN.search(x.name).group():x for x in tqdm(output_tar.getmembers(), 'read output members') if '.json' in x.name}
    
    json_data = {}
    none_data = []
    all_not_matched = []
    steps = 0
    for stock in tqdm(index_dict, "read and match file..."):
        if stock[:2] != '00' and stock[:2] != '60':
            continue
        f2 = output_tar.extractfile(output_dict[stock])
        if f2:
            try:
                tmp_output = json.load(f2)
            except:
                none_data.append(f2.read())
        tmp_output = deal_with_output( extract_year(tmp_output, year, TIME) )
        
        if len(tmp_output)==0:
            continue

        f1 = index_tar.extractfile(index_dict[stock])
        if f1:
            try:
                tmp_index = json.load(f1)
            except:
                none_data.append(f1.read())
        
        tmp_index, invald = deal_with_index(tmp_index)
        result, not_matched = match(tmp_output, tmp_index, stock)
        all_not_matched.extend(not_matched)
        json_data.update(result)

        steps += 1
        if steps % 500 == 0:
            save_data(json_data, '{}_{}'.format(year, steps))
    
    print('finish with datasize: {}'.format(len(json_data)))
    print('unmatched data size: {}; accounted for valid output: {:.4f}'.format(len(all_not_matched),len(all_not_matched)/len(json_data)))

    save_data(json_data, year)
    return json_data, none_data

def deal_with_output(data):
    result = {}
    for t in data:#tqdm(data, 'dealing with output data'):
        if 'title' not in t:
            continue
        if CONTENTS not in t:
            result[t[URL]] = {
                TITLE: t[TITLE], 
                CONTENTS: '',
                LIKE_COUNT: t[LIKE_COUNT],
                TIME: t[TIME],
                }
        else:
            result[t[URL]] = {
                TITLE: t[TITLE], 
                CONTENTS: t[CONTENTS],
                LIKE_COUNT: t[LIKE_COUNT],
                TIME: t[TIME],
                }
    return result

def extract_year(data, year, key_name):
    # 提取出对应年份的data
    valid_data = []
    for x in data:
        if key_name not in x:
            continue
        t = x[key_name]
        t_year = int(t) if len(t)==4 else int(t[:4])
        if t_year == year:
            valid_data.append(x)
    return valid_data

def deal_with_index(data):
    result = {}
    invalid = []
    for t in data:#tqdm(data, 'dealing with index data'):
        if AUTHOR not in t:
            invalid.append(t)
            continue
        uid = t[AUTHOR].split('/')[-1]
        result[t[POST]] = {
            AUTHOR: uid, 
            POST_TYPE: t[POST_TYPE], 
            REPONSE_COUNT: t[REPONSE_COUNT],
            VIEW_COUNT: t[VIEW_COUNT],
            }
    return result,invalid

def read_tar_file(filename):
    tar = tarfile.open(filename,encoding='utf-8')
    json_data = {}
    none_data = {}
    print('reading tar file from {}'.format(filename))
    for x in tqdm(tar.getmembers()):
        name = x.name.split('/')
        if len(name) > 1 and (name[1][:2] =='60' or name[1][:2] =='00'):
            stock_id = name[1][:6]
            f=tar.extractfile(x)
            if f:
                try:
                    json_data[stock_id] = json.load(f)
                except:
                    none_data[stock_id] = f.read()
    print('finish')
    return json_data, none_data

def match(content_data, index_data, stock_id):
    result = {}
    invalid = []
    for key in content_data: #tqdm(content_data, 'match index and content'):
        if key not in index_data:
            invalid.append(key)
            continue
        t1 = content_data[key]
        t2 = index_data[key]
        result[key] = {
            STOCK: stock_id,
            TITLE: t1[TITLE], 
            AUTHOR: t2[AUTHOR], 
            CONTENTS: t1[CONTENTS],
            LIKE_COUNT: t1[LIKE_COUNT],
            TIME: t1[TIME],
            POST_TYPE: t2[POST_TYPE], 
            REPONSE_COUNT: t2[REPONSE_COUNT],
            VIEW_COUNT: t2[VIEW_COUNT],
            }
    return result, invalid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str, default = './data/INDEX.tar.gz', help='file path of index data.')
    parser.add_argument('--output_file', type=str, default = './data/OUTPUT.tar.gz', help='file path of output data.')
    parser.add_argument('--year', type=int, default='2017', help='the year you want to extract data from.')
    args = parser.parse_args()

    json_data, none_data = read_and_match_tar_file(args.index_file, args.output_file, args.year)