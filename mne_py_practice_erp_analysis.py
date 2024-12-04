import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne import Epochs, pick_types
from mne.io import read_raw_brainvision
from mne.datasets import fetch_fsaverage
import os

# まず特定のフォルダに移動
target_directory = '/Users/souta/Documents/Aoyama_Lab/test_program'

# フォルダが存在するか確認し、存在しない場合は作成
if not os.path.isdir(target_directory):
    os.makedirs(target_directory)

# 移動
os.chdir(target_directory)
print(f"Changed working directory to: {os.getcwd()}")

# 作成したいディレクトリのパスリスト（相対パス）
dirs_to_create = ['preprocessed', 'epoch', 'epoch_ica', 'epoch_adjusted', 'epoch_accepted', 'epoch_use', 'epoch_cong_ica']

# 各ディレクトリについて存在チェックを行い、存在しない場合のみ作成
for dir_name in dirs_to_create:
    dir_path = os.path.join(os.getcwd(), dir_name)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")


# Paths to your data
# vhdr_pathsにはディレクトリパスのみを指定
vhdr_paths = ['/Users/souta/Documents/Aoyama_Lab/test_program/sub-101/ses-01/eeg/']
# vhdr_filesにはファイル名のみを指定
vhdr_files = ['sub-101_ses-01_task-objrecog_eeg.vhdr']
#ced_file = './shosha_chanloc.ced'  # This file should be saved manually for the first time.
tsv_file = '/Users/souta/Documents/Aoyama_Lab/drive-download-20240911T021238Z-001/standard_1005.elc'  # This is a commonly used electrode location file for actiCHamp.

# For storing preprocessed, epoch, ica, and adjusted data
setpath_preprocessed = './preprocessed'
setpath_epoch = './epoch'
setpath_epoch_accepted = './epoch_accepted'
setpath_epoch_ica = './epoch_ica'
setpath_epoch_adjusted = './epoch_adjusted'
setpath_epoch_use = './epoch_use'
setpath_epoch_cong_ica = './epoch_cong_ica'

# File paths for the preprocessed, epoch, and adjusted data
setfiles_preprocessed = ['sub_101_week4_stroop_after_preprocessed.fif']
setfiles_epoch_cong = ['sub_101_week4_stroop_after_cong-epo.fif']
setfiles_epoch_incong = ['sub_101_week4_stroop_after_incong-epo.fif']
setfiles_epoch_cong_accepted = ['sub_101_week4_stroop_after_cong_accepted-epo.fif']
setfiles_epoch_incong_accepted = ['sub_101_week4_stroop_after_incong_accepted-epo.fif']
setfiles_epoch_cong_ica = ['sub_101_week4_stroop_after_cong_ica-epo.fif']
setfiles_epoch_incong_ica = ['sub_101_week4_stroop_after_incong_ica-epo.fif']
setfiles_epoch_cong_adjusted = ['sub_101_week4_stroop_after_cong_adjusted-epo.fif']
setfiles_epoch_incong_adjusted = ['sub_101_week4_stroop_after_incong_adjusted-epo.fif']
setfiles_epoch_cong_use = ['sub_101_week4_stroop_after_cong_use-epo.fif']
setfiles_epoch_incong_use = ['sub_101_week4_stroop_after_incong_use-epo.fif']

for sub in range(len(vhdr_files)):
    
    # BrainVision形式の生データを読み込み、前処理を行う
    raw = read_raw_brainvision(os.path.join(vhdr_paths[sub], vhdr_files[sub]), preload=True)
    # カスタムモンタージュを読み込む
    montage = mne.channels.read_custom_montage(tsv_file)
    # 1Hzから100Hzの範囲でフィルタリングを行う
    raw.filter(1, 100, fir_design='firwin', skip_by_annotation='edge')
    # 不良チャンネルを補間する
    raw.interpolate_bads(reset_bads=True)
    # 前処理されたデータを保存する
    raw.save(os.path.join(setpath_preprocessed, setfiles_preprocessed[sub]), overwrite=True)
    #events = mne.find_events(raw, stim_channel='/Users/souta/Documents/Aoyama_Lab/test_program/sub-101/ses-01/eeg/sub-101_ses-01_task-objrecog_events.tsv')
    df = pd.read_csv('/Users/souta/Documents/Aoyama_Lab/test_program/sub-101/ses-01/eeg/sub-101_ses-01_task-objrecog_events.tsv', sep='\t')

    # イベント情報（例えば、'onset'と'trial_type'カラムを使う場合）
    onsets = df['onset'].values  # イベントのオンセットタイム
    event_ids = df['trial_type'].values  # イベントのID

    # サンプリング周波数を取得
    sfreq = raw.info['sfreq']  # サンプリング周波数

    # 時間をサンプル単位に変換
    event_times = (onsets * sfreq).astype(int)

    # イベントのNumPy配列を作成
    events = np.column_stack((event_times, np.zeros_like(event_times), event_ids))

    # イベント情報を直接アノテーションとして追加
    annotations = mne.Annotations(onset=events[:, 0] / raw.info['sfreq'],  # 秒単位のオンセット
                                  duration=np.zeros(len(events)),  # イベントの長さ（全て0に設定）
                                  description=[str(event_id) for event_id in events[:, 2]])  # イベントID

    # 生データにアノテーションを追加
    raw.set_annotations(annotations)

    # 前処理されたデータを保存
    raw.save(os.path.join(setpath_preprocessed, setfiles_preprocessed[sub]), overwrite=True)
    
    print('Preprocessing and saving completed for subject', sub)
    print('example', event_ids)
    #エポッキング
    # データフレームの読み込み
    df = pd.read_csv('/Users/souta/Documents/Aoyama_Lab/test_program/sub-101/ses-01/eeg/sub-101_ses-01_task-objrecog_events.tsv', sep='\t')

    # 'trial_type' をユニークな整数にマッピング
    # trial_type ごとに一意の整数を割り当てる
    event_id = {trial_type: idx for idx, trial_type in enumerate(df['trial_type'].unique())}

    # イベントIDを 'trial_type' から取得
    event_ids = np.array([event_id[trial_type] for trial_type in df['trial_type']])

    # イベントのオンセット時刻（秒単位）をサンプル単位に変換
    sfreq = 1000  # サンプリング周波数（例として1000Hzを使用）
    onsets = df['onset'].values
    event_times = (onsets * sfreq).astype(int)

    # イベント情報をNumPy配列として作成
    events = np.column_stack((event_times, np.zeros_like(event_times), event_ids))

    # イベントの確認
    print(f"Event IDs: {event_ids}")
    print(f"Events: {events}")

    # 生データを読み込み（例: rawデータがあると仮定）
    # raw = mne.io.read_raw_fif('your_raw_data.fif', preload=True)

    # アノテーションを作成
    annotations = mne.Annotations(
        onset=events[:, 0] / sfreq,  # 秒単位でオンセット
        duration=np.zeros(len(events)),  # 全てのイベントの長さは0
        description=[str(event) for event in events[:, 2]]  # イベントIDを文字列に変換
    )

    # アノテーションをrawデータにセット
# raw.set_annotations(annotations)

# これでevent_idとして 'trial_type' を整数で使用することができます

    # Epochs作成（特定のイベントIDを使わず、全イベントを使う場合）
    tmin = -0.2  # 開始時間（秒）
    tmax = 1.0   # 終了時間（秒）

    epochs = mne.Epochs(
    raw, 
    events=events,  # 先ほど作成したeventsを使用
    event_id=None,  # ここで特定のイベントIDを指定せず、全イベントを使う
    tmin=tmin, 
    tmax=tmax, 
    preload=True
)

    epochs.save(os.path.join(setpath_epoch, setfiles_epoch_cong[sub]), overwrite=True)
    print('Epoching completed for subject', sub)

   # Epoch rejection and saving for congruent condition only
for sub in range(len(vhdr_files)):
    # Load the preprocessed epochs data for congruent condition
    epochs_cong = mne.read_epochs(os.path.join(setpath_epoch, setfiles_epoch_cong[sub]))

    # Automatically reject epochs based on threshold (150 µV) for congruent condition
    epochs_cong.drop_bad(reject=dict(eeg=150e-6))  # Reject epochs with >150 µV in EEG channels

    # Check if the 'epochs_accepted' directory exists, create it if it doesn't
    accepted_dir = setpath_epoch_accepted
    if not os.path.exists(accepted_dir):
        os.makedirs(accepted_dir)  # Create the directory if it doesn't exist
        print(f"Created directory: {accepted_dir}")

    # Save accepted epochs for congruent condition to the new directory
    epochs_cong.save(os.path.join(accepted_dir, setfiles_epoch_cong_accepted[sub]), overwrite=True)

    print(f"Epoch rejection and saving for congruent condition completed for subject {sub}")

# ICAの実行と保存
for sub in range(len(vhdr_files)):
    # コンゴ条件の受け入れられたエポックデータを読み込む
    epochs_cong_accepted = mne.read_epochs(os.path.join(setpath_epoch_accepted, setfiles_epoch_cong_accepted[sub]))
    
    print(f"Number of epochs: {len(epochs_cong_accepted)}")
    print(f"Events in epochs: {epochs_cong_accepted.events}")

    # ICAを実行する
    ica_cong = ICA(n_components=20, random_state=97, max_iter=800)
    ica_cong.fit(epochs_cong_accepted)  # ICAを実行するにはepochsオブジェクトを渡す

    # ICAデータを保存する
    ica_cong.save(os.path.join(setpath_epoch_ica, setfiles_epoch_cong_ica[sub]), overwrite=True)

    try:
        print(f"ICA for congruent condition completed and saved for subject {sub}")
        print(f"Event ID: {event_id}")
    except ValueError as e:
        print(f"エラーが発生しました: {e}")
        print("イベントデータが見つかりませんでした。")

# ICAを適用してコンポーネントを除去し、エポックデータを保存する処理
for sub in range(len(vhdr_files)):
    # ICA適用後のエポックデータを読み込む
    try:
        epochs_cong_ica = mne.read_epochs(os.path.join(setpath_epoch_ica, setfiles_epoch_cong_ica[sub]))
    except ValueError as e:
        print(f"エラーが発生しました: {e}")
        print(f"サブジェクト {sub} のエポックデータが見つかりませんでした。スキップします。")
        continue  # エポックデータが見つからない場合は次のサブジェクトに進む

    # 保存したICAオブジェクトを読み込む
    ica_cong = ICA.load(os.path.join(setpath_epoch_ica, setfiles_epoch_cong_ica[sub]))

    # ICAコンポーネントをプロットして不良コンポーネントを確認
    ica_cong.plot_components()  # 目視で不良コンポーネントを確認

    # 除去するコンポーネントを設定（例: コンポーネント0と2を除去）
    ica_cong.exclude = [0, 2]

    # ICAを適用してコンポーネントを除去したエポックデータを作成
    epochs_cong_adjusted = ica_cong.apply(epochs_cong_ica)  # ICAを適用してエポックデータから不良コンポーネントを除去

    # 除去後のエポックデータを保存
    epochs_cong_adjusted.save(os.path.join(setpath_epoch_adjusted, setfiles_epoch_cong_adjusted[sub]), overwrite=True)

    print(f"Adjusted epochs saved for subject {sub}")

ica_cong.plot_drop_log()

"""for sub in range(len(vhdr_files)):
    plot_data = mne.read_epochs(os.path.join(setpath_epoch_ica, setfiles_epoch_cong_ica[sub]))
    plot_data.plot()"""
