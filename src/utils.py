import numpy as np

def intersection(L1_pts, L2_pts):

    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C
    
    L1 = line(L1_pts[0], L1_pts[1])
    L2 = line(L2_pts[0], L2_pts[1])

    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return np.array([x,y])
    else:
        return False
    
'''
Calculate the corners for the quad that is approximately enclosing the contour
'''
def get_corner_pixels(closed_contour, use_4_direction=False):

    # Calculate the 8 cardinal points to find an octogonal shape that is approximately enclosing the label
    cardinal_points = []
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            if i == 0 and j == 0:
                continue
            
            multiplier = [[i],[j]]
            max_point = [[0,0]]
            max_value = float('-inf')

            for p in closed_contour:
                value = p @ multiplier
                if value > max_value:
                    max_point = p
                    max_value = value
            
            cardinal_points.append(max_point)

    #[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]
    # Change the indices to get a convex shape
    octogonal_indices = [6, 7, 4, 2, 1, 0, 3, 5]
    cardinal_points = np.array(cardinal_points)[octogonal_indices]

    # Calculate the lengths of each edge (octogonal shape)
    side_lengths = np.sqrt(np.sum(np.square(np.roll(cardinal_points, -1, axis=0) - cardinal_points), axis=2))

    # Find the largest 4 edges and the points that determine them
    longest_sides = np.sort(np.argpartition(side_lengths.T[0], -4)[-4:])
    side_point_indices = np.array([longest_sides, (longest_sides+1)%8]).T
    side_points = cardinal_points[side_point_indices].reshape(4,2,2)

    # Get the enclosing quad by taking the intersections of the lines on octogonal shape
    # Optional corner points gets only the 4 points (-x-y, -x+y, x-y, x+y).
    corner_points = []; corner_points_opt = []
    for i in range(4):
        corner_points.append([intersection(side_points[i]/100, side_points[(i+1)%4]/100)*100])
        corner_points_opt.append(cardinal_points[2*i+1])

    # Change the indices of elements such that the first-to-second element edge is the largest edge of the quad 
    # (so that later we can apply perspective transform)
    quad_side_lengths = np.sqrt(np.sum(np.square(np.roll(corner_points, -1, axis=0) - corner_points), axis=2)).reshape(4)
    corner_points = np.roll(corner_points, -np.argmax(quad_side_lengths), axis=0)

    quad_side_lengths_opt = np.sqrt(np.sum(np.square(np.roll(corner_points_opt, -1, axis=0) - corner_points_opt), axis=2)).reshape(4)
    corner_points_opt = np.roll(corner_points_opt, -np.argmax(quad_side_lengths_opt), axis=0)

    return (np.array(corner_points, dtype=int) if not use_4_direction else np.array(corner_points_opt, dtype=int))