import numpy as np
import glob
import datetime
import math
import random
import os
import shutil
import matplotlib.pyplot as plt
import pretty_midi
from pypianoroll import Multitrack, Track
import librosa.display
from utils import *

ROOT_PATH = '/Users/sumuzhao/Downloads/'
test_ratio = 0.1
LAST_BAR_MODE = 'remove'


def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % 64) is not 0:
        if LAST_BAR_MODE == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((64 - piano_roll.shape[0] % 64, 128))), axis=0)
        elif LAST_BAR_MODE == 'remove':
            piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0] % 64):], axis=0)
    piano_roll = piano_roll.reshape(-1, 64, 128)
    return piano_roll


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


"""1. divide the original set into train and test sets"""
l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/pop/pop_midi'))]
print(l)
idx = np.random.choice(len(l), int(test_ratio * len(l)), replace=False)
print(len(idx))
for i in idx:
  shutil.move(os.path.join(ROOT_PATH, 'MIDI/pop/pop_midi', l[i]),
              os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/origin_midi', l[i]))


"""2. convert_clean.py"""
from __future__ import print_function
import os
import json
import errno
from pypianoroll import Multitrack, Track
import pretty_midi
import shutil

ROOT_PATH = '/Users/sumuzhao/Downloads/'
converter_path = os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/converter')
cleaner_path = os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner')


def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path does not
    exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_midi_path(root):
    """Return a list of paths to MIDI files in `root` (recursively)"""
    filepaths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.mid'):
                filepaths.append(os.path.join(dirpath, filename))
    return filepaths


def get_midi_info(pm):
    """Return useful information from a pretty_midi.PrettyMIDI instance"""
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time
    else:
        first_beat_time = pm.estimate_beat_start()

    tc_times, tempi = pm.get_tempo_changes()

    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = None

    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'time_signature': time_sign,
        'tempo': tempi[0] if len(tc_times) == 1 else None
    }

    return midi_info


# def midi_filter(midi_info):
#     """Return True for qualified midi files and False for unwanted ones"""
#     if midi_info['first_beat_time'] > 0.0:
#         return False
#     elif midi_info['num_time_signature_change'] > 1:
#         return False
#     elif midi_info['time_signature'] not in ['4/4']:
#         return False
#     return True


# def get_merged(multitrack):
#     """Return a `pypianoroll.Multitrack` instance with piano-rolls merged to
#     five tracks (Bass, Drums, Guitar, Piano and Strings)"""
#     category_list = {'Bass': [], 'Drums': [], 'Guitar': [], 'Piano': [], 'Strings': []}
#     program_dict = {'Piano': 0, 'Drums': 0, 'Guitar': 24, 'Bass': 32, 'Strings': 48}
#
#     for idx, track in enumerate(multitrack.tracks):
#         if track.is_drum:
#             category_list['Drums'].append(idx)
#         elif track.program // 8 == 0:
#             category_list['Piano'].append(idx)
#         elif track.program // 8 == 3:
#             category_list['Guitar'].append(idx)
#         elif track.program // 8 == 4:
#             category_list['Bass'].append(idx)
#         else:
#             category_list['Strings'].append(idx)
#
#     tracks = []
#     for key in category_list:
#         if category_list[key]:
#             merged = multitrack[category_list[key]].get_merged_pianoroll()
#             tracks.append(Track(merged, program_dict[key], key == 'Drums', key))
#         else:
#             tracks.append(Track(None, program_dict[key], key == 'Drums', key))
#    return Multitrack(None, tracks, multitrack.tempo, multitrack.downbeat, multitrack.beat_resolution, multitrack.name)


def converter(filepath):
    """Save a multi-track piano-roll converted from a MIDI file to target
    dataset directory and update MIDI information to `midi_dict`"""
    try:
        midi_name = os.path.splitext(os.path.basename(filepath))[0]
        multitrack = Multitrack(beat_resolution=24, name=midi_name)

        pm = pretty_midi.PrettyMIDI(filepath)
        midi_info = get_midi_info(pm)
        multitrack.parse_pretty_midi(pm)
        # merged = get_merged(multitrack)
        merged = multitrack

        make_sure_path_exists(converter_path)
        merged.save(os.path.join(converter_path, midi_name + '.npz'))

        return [midi_name, midi_info]

    except:
        return None


def main():
    """Main function of the converter"""
    midi_paths = get_midi_path(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/origin_midi'))
    midi_dict = {}
    kv_pairs = [converter(midi_path) for midi_path in midi_paths]
    for kv_pair in kv_pairs:
        if kv_pair is not None:
            midi_dict[kv_pair[0]] = kv_pair[1]

    with open(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/midis.json'), 'w') as outfile:
        json.dump(midi_dict, outfile)

    print("[Done] {} files out of {} have been successfully converted".format(len(midi_dict), len(midi_paths)))

    with open(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/midis.json')) as infile:
        midi_dict = json.load(infile)
    count = 0
    make_sure_path_exists(cleaner_path)
    midi_dict_clean = {}
    for key in midi_dict:
        # if midi_filter(midi_dict[key]):
            midi_dict_clean[key] = midi_dict[key]
            count += 1
            shutil.copyfile(os.path.join(converter_path, key + '.npz'),
                            os.path.join(cleaner_path, key + '.npz'))

    with open(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/midis_clean.json'), 'w') as outfile:
        json.dump(midi_dict_clean, outfile)

    print("[Done] {} files out of {} have been successfully cleaned".format(count, len(midi_dict)))


if __name__ == "__main__":
    main()

"""3. choose the clean midi from original sets"""
if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi')):
  os.makedirs(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi'))
l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner'))]
print(l)
print(len(l))
for i in l:
  shutil.copy(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/origin_midi', os.path.splitext(i)[0] + '.mid'),
              os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi', os.path.splitext(i)[0] + '.mid'))

"""4. merge and crop"""
if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi_gen')):
    os.makedirs(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi_gen'))
if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy')):
    os.makedirs(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy'))
l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi'))]
print(l)
count = 0
for i in range(len(l)):
    try:
        multitrack = Multitrack(beat_resolution=4, name=os.path.splitext(l[i])[0])
        x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi', l[i]))
        multitrack.parse_pretty_midi(x)

        category_list = {'Piano': [], 'Drums': []}
        program_dict = {'Piano': 0, 'Drums': 0}

        for idx, track in enumerate(multitrack.tracks):
            if track.is_drum:
                category_list['Drums'].append(idx)
            else:
                category_list['Piano'].append(idx)
        tracks = []
        merged = multitrack[category_list['Piano']].get_merged_pianoroll()
        print(merged.shape)

        pr = get_bar_piano_roll(merged)
        print(pr.shape)
        pr_clip = pr[:, :, 24:108]
        print(pr_clip.shape)
        if int(pr_clip.shape[0] % 4) != 0:
            pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
        pr_re = pr_clip.reshape(-1, 64, 84, 1)
        print(pr_re.shape)
        save_midis(pr_re, os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi_gen', os.path.splitext(l[i])[0] +
                                       '.mid'))
        np.save(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy', os.path.splitext(l[i])[0] + '.npy'), pr_re)
    except:
        count += 1
        print('Wrong', l[i])
        continue
print(count)

"""5. concatenate into a big binary numpy array file"""
l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy'))]
print(l)
train = np.load(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy', l[0]))
print(train.shape, np.max(train))
for i in range(1, len(l)):
    print(i, l[i])
    t = np.load(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy', l[i]))
    train = np.concatenate((train, t), axis=0)
print(train.shape)
np.save(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/pop_test_piano.npy'), (train > 0.0))

"""6. separate numpy array file into single phrases"""
if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/phrase_test')):
    os.makedirs(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/phrase_test'))
x = np.load(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/pop_test_piano.npy'))
print(x.shape)
count = 0
for i in range(x.shape[0]):
    if np.max(x[i]):
        count += 1
        np.save(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/phrase_test/pop_piano_test_{}.npy'.format(i+1)), x[i])
        print(x[i].shape)
   # if count == 11216:
   #     break
print(count)

"""some other codes"""
filepaths = []
msd_id_list = []
for dirpath, _, filenames in os.walk(os.path.join(ROOT_PATH, 'MIDI/Sinfonie Data')):
    for filename in filenames:
        if filename.endswith('.mid'):
            msd_id_list.append(filename)
            filepaths.append(os.path.join(dirpath, filename))
print(filepaths)
print(msd_id_list)
for i in range(len(filepaths)):
    shutil.copy(filepaths[i], os.path.join(ROOT_PATH, 'MIDI/classic/classic_midi/{}'.format(msd_id_list[i])))

x1 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_1.npy'))
x2 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_2.npy'))
x3 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_3.npy'))
x4 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_4.npy'))
x5 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_5.npy'))
x = np.concatenate((x1, x2, x3, x4, x5), axis=0)
print(x.shape)
np.save(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano.npy'), x)


multitrack = Multitrack(beat_resolution=4, name='YMCA')
x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/origin/YMCA.mid'))
multitrack.parse_pretty_midi(x)

category_list = {'Piano': [], 'Drums': []}
program_dict = {'Piano': 0, 'Drums': 0}

for idx, track in enumerate(multitrack.tracks):
    if track.is_drum:
        category_list['Drums'].append(idx)
    else:
        category_list['Piano'].append(idx)
tracks = []
merged = multitrack[category_list['Piano']].get_merged_pianoroll()

# merged = multitrack.get_merged_pianoroll()
print(merged.shape)

pr = get_bar_piano_roll(merged)
print(pr.shape)
pr_clip = pr[:, :, 24:108]
print(pr_clip.shape)
if int(pr_clip.shape[0] % 4) != 0:
    pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
pr_re = pr_clip.reshape(-1, 64, 84, 1)
print(pr_re.shape)
save_midis(pr_re, os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/merged_midi/YMCA.mid'), 127)
np.save(os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/merged_npy/YMCA.npy'), (pr_re > 0.0))
