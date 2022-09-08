import numpy as np
import pandas as pd

class BilateralSymmetry():
    """The algorithm generates a symmetry score based on cell dispersion in the organ of interest
    The algorithm works together with ImageJ to obtain X and Y coordinates of each cell
    The values of symmetry are between 0 and 1, where 1 is a complete perfect symmetry and 0 shows non-symmetric relationship between hemispheres of the organ"""

    def __init__(self, data_f, symmetry_temp, find=False):
        self.y_sym = 0
        self.df = pd.read_csv(data_f, delimiter='\t', names=['x', 'y'])
        self.df_left = self.df[self.df['y'] < symmetry_temp]
        self.df_right = self.df[self.df['y'] > symmetry_temp]
        self.x_sym = self.df['x'].min()
        self.transform = np.array([1,-1])
        if find == False:
            self.y_sym = symmetry_temp
        else:
            self.y_sym = (self.df_left['y'].max() + self.df_right['y'].min()) / 2

    def Mapping(self):
        """""The function maps each cell both sides to its opposite side"""
        b = np.array([self.x_sym, self.y_sym])
        c = self.transform
        self.mapped_side_left = (self.df_left - b) * c + b # this is the left hemisphere mirrored to the right hemisphere
        self.mapped_side_right = (self.df_right - b) * c + b # this is the right hemisphere mirrored to the left hemisphere

        return (self.mapped_side_left, self.mapped_side_right)

    def Symmetric_score(self):
        """It yields a symmetric score based on a coefficient of determination
        It will also return both hemispheres paired arrays"""

        # substract the one side of the other to and get the vector magnitud to compare distance
        points_evaluated_0 = []
        closest_points_real = []  # it will contain the indexes of the closest points

        for i in self.df_right.index:
            a = ((self.mapped_side_left.values - self.df_right.loc[i].values) ** 2).sum(axis=1, keepdims=True)
            points_evaluated_0.append(np.where(a[:, 0] == a[:, 0].min())[0][0] + self.mapped_side_left.index[0])
            closest_points_real.append((points_evaluated_0[i], i))  # pairs of rows of closest cells

        one = set(self.mapped_side_left.index)
        two = set(np.array(points_evaluated_0))
        difference = list(one.difference(two))
        rest_of_points = []
        # pairing the rest of the points
        for i in list(difference):
            a = ((self.df_right.values - self.mapped_side_left.loc[i].values) ** 2).sum(axis=1, keepdims=True)
            rest_of_points.append((i, (np.where(a[:, 0] == a[:, 0].min())[0][
                0])))  # pairs rows of closest cells mirrored hemisphere and normal hemisphere

        all_pairs = sorted(closest_points_real + rest_of_points)  # all pairs of closest cells
        all_pairs_right_mirror_x = np.array([self.mapped_side_left.loc[i] for i in np.array(all_pairs)[:, 0]])
        all_pairs_df_right_y = np.array([self.df_right.loc[i] for i in np.array(all_pairs)[:, 1]])

        y = all_pairs_df_right_y
        x = all_pairs_right_mirror_x

        a = (((y[:, 0] - x[:, 0]) ** 2).sum() + ((y[:, 1] - x[:, 1]) ** 2).sum())
        b = (((y[:, 0] - y[:, 0].mean()) ** 2).sum() + ((y[:, 1] - y[:, 1].mean()) ** 2).sum())
        symmetry_score = (b - a) / b

        return max(0, symmetry_score), x, y
