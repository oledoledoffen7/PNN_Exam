import numpy as np
import sys
import copy

data = np.array([(-1, 3), (1, 2), (0, 1), (4, 0), (5, 4), (3, 2)])
cluster_centre_1 = np.array([(-1, 3)])
cluster_centre_2 = np.array([(3, 2)])
cluster_centres = [cluster_centre_1, cluster_centre_2]

def k_means_clustering(data, cluster_centres):
    clusters_not_changed = False
    while clusters_not_changed == False:
        previous_cluster_centres = copy.copy(cluster_centres)
        assigned_clusters = []
        for i in range(len(data)):
            closest_cluster = None
            closest_cluster_distance = sys.maxsize
            for j in range(len(cluster_centres)):
                distance_to_cluster = np.linalg.norm(data[i] - cluster_centres[j]) 
                if closest_cluster == None:
                    closest_cluster = j 
                    closest_cluster_distance = distance_to_cluster
                elif distance_to_cluster < closest_cluster_distance:
                    closest_cluster = j    
                    closest_cluster_distance = distance_to_cluster
            assigned_clusters.append((data[i], closest_cluster))
        for j in range(len(cluster_centres)):
            cluster_centres[j] = None
            number_of_points_assigned = 0
            for k in range(len(assigned_clusters)):
                if assigned_clusters[k][1] == j:
                    number_of_points_assigned = number_of_points_assigned + 1
                    if number_of_points_assigned == 1:
                        cluster_centres[j] = assigned_clusters[k][0]
                    else:
                        cluster_centres[j] = cluster_centres[j] + assigned_clusters[k][0]
            cluster_centres[j] = cluster_centres[j] / number_of_points_assigned
        if np.array_equal(previous_cluster_centres, cluster_centres) == True:
            clusters_not_changed = True
    return cluster_centres

result = k_means_clustering(data, cluster_centres)
print(result)