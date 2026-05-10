def simple_cluster(data, threshold=1.0):
    """
    Groups data into clusters where the distance between consecutive 
    elements is less than the threshold.
    
    Args:
        data (list): A list of numerical values.
        threshold (float): The distance threshold to start a new cluster.
        
    Returns:
        list: A list of lists, where each inner list is a cluster.
    """
    if not data:
        return []
    
    sorted_data = sorted(data)
    clusters = [[sorted_data[0]]]
    
    for val in sorted_data[1:]:
        if val - clusters[-1][-1] <= threshold:
            clusters[-1].append(val)
        else:
            clusters.append([val])
            
    return clusters
