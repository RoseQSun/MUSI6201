import json
import requests
import pandas as pd
import librosa
import soundfile as sf

base_url = 'http://musicbrainz.org/ws/2/recording/?fmt=json&limit=1&query='


# resp = requests.get('http://musicbrainz.org/ws/2/recording/?fmt=json&limit=1&query=《秦香莲》')
# data = json.loads(resp.text)
# data

# df =  pd.DataFrame(columns=['File', 'Class1', 'Class2'])

# tags = data['recordings'][0]['tags']

# if (len(tags) == 2):
#     song = {'File': 'file', 'Class1': tags[0]['name'], 'Class2': tags[1]['name']}
#     df = df.append(song, ignore_index=True)


# df.to_csv(r'/Users/jocekav/Documents/GitHub/AudioContentAnalysis/df.csv', index=False)

import os
import time

df =  pd.DataFrame(columns=['File', 'Class1', 'Class2'])
for (root, dirs, files) in os.walk('/Volumes/Samsung USB/JingjuAudioRecordingsCollection'):
    for file in files:
        if file.endswith('.flac'):
            name = file.split('-')[-1]
            name = name[:len(name)-5]
            resp = requests.get(base_url + name)
            data = json.loads(resp.text)
            try:
                tags = data['recordings'][0]['tags']
            except:
                print(data)
            if (len(tags) == 2):
                song = {'File': file, 'Class1': tags[0]['name'], 'Class2': tags[1]['name']}
                df = df.append(song, ignore_index=True)
            time.sleep(1)

df.to_csv('/Users/jocekav/Documents/GitHub/AudioContentAnalysis/Tagged_Data.csv', index=False)

df = pd.read_csv ('Tagged_Data.csv')

types = ['manban', 'zhongsanyan', 'kuaisanyan', 'yuanban', 'erliu', 'liushui', 'kuaiban', 'sanban', 'yaoban']
for type in types:
    df1 = df[df['Class1'].str.contains('kuaisanyan')]
    df2 = df[df['Class2'].str.contains('kuaisanyan')]
    total_df = pd.concat([df1, df2])
    file_name = '/Users/jocekav/Documents/GitHub/AudioContentAnalysis/' + 'kuaisanyan' + '_data.csv'
    total_df.to_csv(file_name, index=False)

data, fs = sf.read('/Volumes/Samsung USB/JingjuAudioRecordingsCollection/中国京剧名票系列专辑之三 阮宝利京剧唱腔选/01 - 阮宝利  -  《秦香莲》 （夫在东来妻在西）.flac')
print(data)



