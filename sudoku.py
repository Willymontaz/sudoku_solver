#!/usr/bin/python3

import json
from copy import deepcopy

default_matrix = [
    [
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
    ],
    [
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
    ],
    [
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
    ]
]


class sudoku(object):
    def __init__(self, *args, **kwargs):
        json_file = kwargs.get('json_file', None)
        if len(args) > 0:
            self.load(args[0])
        elif json_file != None:
            self.load(file)
        else:
            # Init matrix
            self.matrix = default_matrix

    def is_valid(self, ):
        ret_list = list()

        # Check lines
        for num_line in range(9):
            num_values = self.get_line(num_line)
            unique_values = list(set(num_values))
            if 0 in unique_values:
                unique_values.remove(0)

            # list value found twice
            twice_list = list()
            for tmp_value in unique_values:
                if (num_values.count(tmp_value)) > 1:
                    twice_list.append(tmp_value)

            # Save tuples
            for num_row in range(9):
                if self.get_value((num_line, num_row)) in twice_list:
                    ret_list.append((num_line, num_row))

        # Check rows
        for num_row in range(9):
            num_values = self.get_row(num_row)
            unique_values = list(set(num_values))
            if 0 in unique_values:
                unique_values.remove(0)

            # list value found twice
            twice_list = list()
            for tmp_value in unique_values:
                if (num_values.count(tmp_value)) > 1:
                    twice_list.append(tmp_value)

            # Save tuples
            for num_line in range(9):
                if self.get_value((num_line, num_row)) in twice_list:
                    ret_list.append((num_line, num_row))

        # Check block
        for x_block in range(3):
            for y_block in range(3):
                num_values = self.flatten_list(self.get_block(x_block, y_block))
                unique_values = list(set(num_values))
                if 0 in unique_values:
                    unique_values.remove(0)

                # list value found twice
                twice_list = list()
                for tmp_value in unique_values:
                    if (num_values.count(tmp_value)) > 1:
                        twice_list.append(tmp_value)

                # Save tuples                
                for row_block in range(3):
                    for line_block in range(3):
                        num_line = x_block * 3 + line_block
                        num_row = y_block * 3 + row_block
                        if self.get_value((num_line, num_row)) in twice_list:
                            ret_list.append((num_line, num_row))

        if len(ret_list) == 0:
            return True, None
        else:
            return False, ret_list

    def load(self, json_file):
        with open(json_file, 'r') as file:
            self.matrix = json.loads(file.read())

    def get_block(self, num_line, num_row):
        return self.matrix[num_line][num_row]

    def get_row(self, num_row):
        tmp_list = list()

        x = int(num_row / 3)
        y = num_row % 3

        for idx in range(3):
            for subitem in self.matrix[idx][x]:
                tmp_list.append(subitem[y])
        return tmp_list

    def get_line(self, num_line):
        tmp_list = list()

        x = int(num_line / 3)
        y = num_line % 3

        for item in self.matrix[x]:
            tmp_list += item[y]
        return tmp_list

    def get_value(self, num_tuple):
        num_line, num_row = num_tuple
        return self.get_line(num_line)[num_row]

    def set_value(self, num_value, num_tuple):
        num_line, num_row = num_tuple
        self.matrix[int(num_line / 3)][int(num_row / 3)][int(num_line % 3)][int(num_row % 3)] = num_value

    def flatten_list(self, list_example):
        ret_list = list()
        for item in list_example:
            if type(item) == list:
                ret_list += self.flatten_list(item)
            else:
                ret_list.append(item)
        return ret_list

    def get_remain_values(self, ):
        flat_list = self.flatten_list(self.matrix)
        ret_dict = dict()
        for x in range(1, 10):
            ret_dict[x] = 9 - flat_list.count(x)
        return ret_dict

    def print_remain_values(self, ):
        print("Remain values:")
        flat_list = self.flatten_list(self.matrix)
        for x in range(1, 10):
            print("  * " + str(x) + " = " + str(9 - flat_list.count(x)))

    def print_matrix(self, ):
        for line in self.matrix:
            for x in range(3):
                print(str(line[0][x]) + "  " + str(line[1][x]) + "  " + str(line[2][x]))
            print("")

    def copy(self, matrix_to_copy):
        for block_line in range(3):
            for block_row in range(3):
                for line in range(3):
                    self.matrix[block_line][block_row][line] = deepcopy(
                        matrix_to_copy.matrix[block_line][block_row][line])


if __name__ == '__main__':
    sudoku()
    sudoku("test")
    sudoku(json_file="test")
