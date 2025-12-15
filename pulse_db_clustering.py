# pulse_db_full.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os

# ------------------------------
# 1. Data Loader
# ------------------------------
class TimeSeriesLoader:
    def __init__(self, file_path=None):
        self.data = None
        if file_path:
            self.data = self.load_csv(file_path)

    def load_csv(self, path):
        """
        Load time-series data from CSV.
        Each row is one 10-second segment.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        return np.loadtxt(path, delimiter=',')


# ------------------------------
# 2. Divide-and-Conquer Clustering
# ------------------------------
class DivideConquerClustering:
    def __init__(self, similarity_threshold=0.9, min_cluster_size=5):
        self.threshold = similarity_threshold
        self.min_size = min_cluster_size

    def cluster(self, segments):
        """
        Recursive divide-and-conquer clustering based on correlation.
        Returns a list of clusters (each cluster is a list of segments)
        """
        if len(segments) <= self.min_size:
            return [segments]

        # Compute similarity to first segment
        first_segment = segments[0]
        sims = [np.corrcoef(first_segment, s)[0, 1] for s in segments]

        cluster1 = [segments[i] for i in range(len(segments)) if sims[i] >= self.threshold]
        cluster2 = [segments[i] for i in range(len(segments)) if sims[i] < self.threshold]

        clusters = []
        if cluster1:
            clusters += self.cluster(cluster1)
        if cluster2:
            clusters += self.cluster(cluster2)

        return clusters


# ------------------------------
# 3. Closest Pair Finder
# ------------------------------
class ClosestPair:
    def __init__(self, metric='DTW'):
        self.metric = metric

    def find_closest(self, cluster):
        """
        Find the closest pair of time series in a cluster using DTW or Euclidean
        """
        min_dist = float('inf')
        pair = (None, None)
        n = len(cluster)
        for i in range(n):
            for j in range(i + 1, n):
                if self.metric == 'DTW':
                    dist, _ = fastdtw(cluster[i], cluster[j], dist=euclidean)
                else:
                    dist = np.linalg.norm(cluster[i] - cluster[j])
                if dist < min_dist:
                    min_dist = dist
                    pair = (cluster[i], cluster[j])
        return pair, min_dist


# ------------------------------
# 4. Maximum Subarray Analyzer
# ------------------------------
class MaxSubarrayAnalyzer:
    def kadane(self, series):
        """
        Kadane's algorithm: returns max sum and interval (start, end)
        """
        max_sum = -float('inf')
        current_sum = 0
        start = end = s = 0
        for i, val in enumerate(series):
            current_sum += val
            if current_sum > max_sum:
                max_sum = current_sum
                start = s
                end = i
            if current_sum < 0:
                current_sum = 0
                s = i + 1
        return max_sum, (start, end)


# ------------------------------
# 5. Report Generator
# ------------------------------
class ReportGenerator:
    def generate_report(self, clusters, closest_pairs, max_intervals, top_n_segments=2):
        print("=== Clustering Report ===")
        print("Number of clusters:", len(clusters))
        for idx, cluster in enumerate(clusters):
            print(f"\nCluster {idx+1}: Size = {len(cluster)}")
            print("Closest pair distance:", closest_pairs[idx][1])
            print("Max subarray intervals for first segment:", max_intervals[idx][0])

            # Plot top_n_segments
            plt.figure(figsize=(8, 4))
            for i in range(min(top_n_segments, len(cluster))):
                plt.plot(cluster[i], label=f'Segment {i+1}')
            plt.title(f'Cluster {idx+1} Example Segments')
            plt.legend()
            plt.show()


# ------------------------------
# 6. Full Execution for 1000 PulseDB segments
# ------------------------------
def run_pulse_db(file_path):
    # Load segments
    loader = TimeSeriesLoader(file_path)
    segments = loader.data.tolist()  # Convert to list for clustering
    print(f"Loaded {len(segments)} segments from {file_path}")

    # Step 1: Cluster
    clustering = DivideConquerClustering(similarity_threshold=0.9, min_cluster_size=5)
    clusters = clustering.cluster(segments)
    print(f"Formed {len(clusters)} clusters")

    # Step 2: Closest pair
    closest_pair_finder = ClosestPair(metric='DTW')
    closest_pairs = []
    for cluster in clusters:
        pair, dist = closest_pair_finder.find_closest(cluster)
        closest_pairs.append((pair, dist))

    # Step 3: Max subarray
    analyzer = MaxSubarrayAnalyzer()
    max_intervals = []
    for cluster in clusters:
        cluster_intervals = [analyzer.kadane(s) for s in cluster]
        max_intervals.append(cluster_intervals)

    # Step 4: Reporting
    reporter = ReportGenerator()
    reporter.generate_report(clusters, closest_pairs, max_intervals, top_n_segments=2)


# ------------------------------
# 7. Main Execution
# ------------------------------
if __name__ == "__main__":
    # Replace with your PulseDB CSV path
    pulse_db_file = "pulse_db_segments.csv"
    if os.path.exists(pulse_db_file):
        run_pulse_db(pulse_db_file)
    else:
        print("PulseDB file not found. Running toy example instead.")
        # Toy Example
        ts1 = np.array([1, 2, 3, 1, 0])
        ts2 = np.array([2, 3, 4, 2, 1])
        ts3 = np.array([10, 11, 10, 9, 8])
        toy_segments = [ts1, ts2, ts3]
        clustering = DivideConquerClustering(similarity_threshold=0.9)
        clusters = clustering.cluster(toy_segments)
        closest_pair_finder = ClosestPair()
        closest_pairs = [closest_pair_finder.find_closest(c) for c in clusters]
        analyzer = MaxSubarrayAnalyzer()
        max_intervals = [[analyzer.kadane(s) for s in c] for c in clusters]
        reporter = ReportGenerator()
        reporter.generate_report(clusters, closest_pairs, max_intervals)
