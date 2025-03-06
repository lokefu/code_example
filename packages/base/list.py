#### get union intervals from a list of intervals
def merge_intervals(intervals):
    merged = []
    for interval in sorted(intervals, key=lambda x: x[0]):
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

#### find duplicates in a list
def find_same_values_indices(lst):
    indices = {}
    for idx, value in enumerate(lst):
        if value in indices:
            indices[value].append(idx)
        else:
            indices[value] = [idx]
    return {k: v for k, v in indices.items() if len(v) > 1}
#{2: [1, 3], 3: [2, 5, 10], 5: [6, 9]}

#### sum of all intervals of a list
def sum_intervals(intervals):
    return sum([end - start for start, end in intervals])

#### average of list
def average(lst):
    return sum(lst) / len(lst)

#### intersection between list and list of list
def find_intersections(pred, truth_list):
    def find_intersection(pred, interval):
        start = max(pred[0], interval[0])
        end = min(pred[1], interval[1])
        if start <= end:
            return [start, end]
        return None

    intersections = [find_intersection(pred, interval) for interval in truth_list]
    intersections = [interval for interval in intersections if interval is not None]

    return intersections
#pred = [3, 10]
#truth_list = [[0, 5.8], [6.2, 20]]

#### union between list and list of list
def find_union(pred, truth_list):
    all_intervals = truth_list + [pred]
    union = merge_intervals(all_intervals)
    return union

# Example usage
#pred = [3, 10]
#truth_list = [[0, 5.8], [6.2, 20]]

#### IoU between list and list of list
def find_IoU(pred, truth_list):
    merged_truth_list = merge_intervals(truth_list)
    intersections = find_intersections(pred, merged_truth_list)
    intersection = sum_intervals(intersections)
    union_list = find_union(pred, merged_truth_list)
    union = sum_intervals(union_list)
    return intersection / union