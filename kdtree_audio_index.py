import os
import numpy as np
import librosa
from scipy.spatial import KDTree
import pickle

class KDTreeAudioIndexer:
    def __init__(self):
        self.audio_map = {}  # id -> filename
        self.vectors = []
        self.tree = None
        self.counter = 0
        self.FRAME_SIZE = 1024
        self.HOP_SIZE = 512

    def extract_features(self, filepath):
        y, sr = librosa.load(filepath, sr=None)
        # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        rms = np.mean(librosa.feature.rms(y=y, 
                                          frame_length=self.FRAME_SIZE, 
                                          hop_length=self.HOP_SIZE)[0])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y, 
                                                         frame_length=self.FRAME_SIZE, 
                                                         hop_length=self.HOP_SIZE)[0])
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, 
                                          n_fft=self.FRAME_SIZE, 
                                          hop_length=self.HOP_SIZE)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, 
                                          n_fft=self.FRAME_SIZE, 
                                          hop_length=self.HOP_SIZE)[0])
        amplitude_envelope = np.mean(self.amplitude_envelope(y, self.FRAME_SIZE, self.HOP_SIZE))
        return [rms, zcr, spectral_centroid, spectral_bandwidth, amplitude_envelope]

    def amplitude_envelope(self, signal, frame_size, hop_length):
        """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
        amplitude_envelope = []
        
        # calculate amplitude envelope for each frame
        for i in range(0, len(signal), hop_length): 
            amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
            amplitude_envelope.append(amplitude_envelope_current_frame)
        
        return np.array(amplitude_envelope)
    
    def index_audio_file(self, filepath, filename):
        features = self.extract_features(filepath)
        self.vectors.append(features)
        self.audio_map[self.counter] = filename
        self.counter += 1
        self.build_tree()
        
    def load_paths(self, data_folder):
        paths = []
        for percusion_dir in os.listdir(data_folder):
            for file in os.listdir(os.path.join(data_folder, percusion_dir)):
                audio_path = data_folder + "/" + percusion_dir + "/" + file
                paths.append(audio_path)
        return paths
        
    def load_data_and_build_tree(self, data_folder):
        print("Start load data to build tree")
        paths = self.load_paths(data_folder)
        for i in range(len(paths)):
            features = self.extract_features(paths[i])
            self.vectors.append(features)
            self.audio_map[i] = paths[i]
        self.counter = len(paths) + 1
        self.build_tree()
        print("Finish load data and build tree")

    def build_tree(self):
        if self.vectors:
            self.tree = KDTree(np.array(self.vectors))

    def query(self, filepath, k=3):
        if not self.tree:
            return []
        query_vec = self.extract_features(filepath)
        distances, indices = self.tree.query(query_vec, k=k, p=1)
        results = [(self.audio_map[i], distances[j]) for j, i in enumerate(indices)]
        return results

