#!/usr/bin/python3

import argparse
from itertools import product

from sudoku import *


################################################################
#
#  ######  ##     ## ########  ######  ##    ##
# ##    ## ##     ## ##       ##    ## ##   ##
# ##       ##     ## ##       ##       ##  ##
# ##       ######### ######   ##       #####
# ##       ##     ## ##       ##       ##  ##
# ##    ## ##     ## ##       ##    ## ##   ##
#  ######  ##     ## ########  ######  ##    ##
#
################################################################
def check_is_num_exist_for_tuple(sudoku_obj, num_value, num_tuple):
    # Get positions
    block_line = int(num_tuple[0] / 3)
    block_row = int(num_tuple[1] / 3)

    # CHECK VALUE
    if 0 != sudoku_obj.get_value(num_tuple):
        return False

    # CHECK BLOCK
    tmp_block = sudoku_obj.flatten_list(sudoku_obj.get_block(block_line, block_row))
    if num_value in tmp_block:
        return False

    # CHECK LINE
    tmp_line = sudoku_obj.get_line(num_tuple[0])
    if num_value in tmp_line:
        return False

    # CHECK ROW
    tmp_row = sudoku_obj.get_row(num_tuple[1])
    if num_value in tmp_row:
        return False

    # ELSE
    return True


def check_is_num_unique(sudoku_obj, tmp_couple, possibility_dict):
    for check_func in [check_unique_in_line, check_unique_in_row, check_unique_in_intersection]:
        tmp_list = list(filter(lambda x: check_func(sudoku_obj, x, tmp_couple), possibility_dict[tmp_couple]))
        if len(tmp_list) == 1:
            possibility_dict[tmp_couple] = tmp_list
            return


def check_unique_in_line(sudoku_obj, num_value, num_tuple):
    x, y = num_tuple
    line = sudoku_obj.get_line(x)

    count_validity = 0
    for row_idx, row_value in enumerate(line):
        if row_idx == y:
            continue
        if check_is_num_exist_for_tuple(sudoku_obj, num_value, (x, row_idx)):
            count_validity += 1

    if count_validity == 0:
        return True
    else:
        return False


def check_unique_in_row(sudoku_obj, num_value, num_tuple):
    x, y = num_tuple
    row = sudoku_obj.get_row(y)

    count_validity = 0
    for line_idx, line_value in enumerate(row):
        if line_idx == x:
            continue
        if check_is_num_exist_for_tuple(sudoku_obj, num_value, (line_idx, y)):
            count_validity += 1

    if count_validity == 0:
        return True
    else:
        return False


def check_unique_in_intersection(sudoku_obj, num_value, num_tuple):
    # Get line & row
    x, y = num_tuple
    line = sudoku_obj.get_line(x)
    row = sudoku_obj.get_row(y)
    block = sudoku_obj.flatten_list(sudoku_obj.get_block(int(x / 3), int(y / 3)))
    tmp_list = set(line + row + block)
    tmp_list.remove(0)

    if len(tmp_list) == 8:
        return True
    else:
        return False


def check_unique_per_block(sudoku_obj, possibility_dict):
    for block_num in range(9):
        block_list = list(filter(lambda x: block_num == int(x[0] / 3) * 3 + int(x[1] / 3), possibility_dict.keys()))

        # Get unique in this block
        block_values = sudoku_obj.flatten_list(list(map(lambda x: possibility_dict[x], block_list)))

        # Unique num
        unique_num = list()
        for num_value in list(set(block_values)):
            if block_values.count(num_value) == 1:
                unique_num.append(num_value)

        # Update possibility list
        for tmp_couple in block_list:
            num_list = list(filter(lambda x: x in unique_num, possibility_dict[tmp_couple]))
            if len(num_list) == 1:
                possibility_dict[tmp_couple] = num_list


################################################################
#
#  ######   ######     ###    ##    ##         ##    ## ##     ## ##     ##
# ##    ## ##    ##   ## ##   ###   ##         ###   ## ##     ## ###   ###
# ##       ##        ##   ##  ####  ##         ####  ## ##     ## #### ####
#  ######  ##       ##     ## ## ## ##         ## ## ## ##     ## ## ### ##
#       ## ##       ######### ##  ####         ##  #### ##     ## ##     ##
# ##    ## ##    ## ##     ## ##   ###         ##   ### ##     ## ##     ##
#  ######   ######  ##     ## ##    ## ####### ##    ##  #######  ##     ##
#
################################################################
def scan_num(sudoku_obj, num_value):
    # Search potential line
    x_list = list()
    for x in range(9):
        if not num_value in sudoku_obj.get_line(x):
            x_list.append(x)

    # Search potential row
    y_list = list()
    for y in range(9):
        if not num_value in sudoku_obj.get_row(y):
            y_list.append(y)

    # Get all couple
    couple_list = list(product(x_list, y_list))

    # Filter couple available
    couple_list = list(filter(lambda x: check_is_num_exist_for_tuple(sudoku_obj, num_value, x), couple_list))

    # Return untreat valid list
    return couple_list


def scan_update(sudoku_obj, possibility_dict):
    delete_list = list()

    # Check values
    for tmp_couple in possibility_dict.keys():
        # Check num validity
        possibility_dict[tmp_couple] = list(
            filter(lambda x: check_is_num_exist_for_tuple(sudoku_obj, x, tmp_couple), possibility_dict[tmp_couple]))
        # Check num is unique
        check_is_num_unique(sudoku_obj, tmp_couple, possibility_dict)
        # Update delete list
        if len(possibility_dict[tmp_couple]) == 0:
            delete_list.append(tmp_couple)

    # Check per block
    check_unique_per_block(sudoku_obj, possibility_dict)

    for tmp_couple in delete_list:
        possibility_dict.pop(tmp_couple)


################################################################
#
# ########  ########   #######   ######  ########  ######   ######  
# ##     ## ##     ## ##     ## ##    ## ##       ##    ## ##    ## 
# ##     ## ##     ## ##     ## ##       ##       ##       ##       
# ########  ########  ##     ## ##       ######    ######   ######  
# ##        ##   ##   ##     ## ##       ##             ##       ## 
# ##        ##    ##  ##     ## ##    ## ##       ##    ## ##    ## 
# ##        ##     ##  #######   ######  ########  ######   ######  
#
################################################################
def process_unique(sudoku_obj, possibility_dict):
    # Create unique list
    unique_list = list()
    for tmp_couple in possibility_dict.keys():
        if len(possibility_dict[tmp_couple]) == 1:
            unique_list.append(tmp_couple)
            sudoku_obj.set_value(possibility_dict[tmp_couple][0], tmp_couple)

    # Update ret_dict
    for tmp_couple in unique_list:
        possibility_dict.pop(tmp_couple)

    if len(unique_list) > 0:
        return True
    else:
        return False


def process_duplicates(sudoku_obj, possibility_dict):
    # Filter duplicates
    filter_duplicates_per_block(sudoku_obj, possibility_dict)
    filter_duplicates_per_line(sudoku_obj, possibility_dict)
    filter_duplicates_per_row(sudoku_obj, possibility_dict)

    if process_duplicates_per_block(sudoku_obj, possibility_dict): return True
    if process_duplicates_per_line(sudoku_obj, possibility_dict): return True
    if process_duplicates_per_row(sudoku_obj, possibility_dict): return True

    return False


def process_duplicates_per_block(sudoku_obj, possibility_dict):
    for block_num in range(9):
        # Filter possibility per block
        couple_list = list(filter(lambda x: block_num == int(x[0] / 3) * 3 + int(x[1] / 3), possibility_dict.keys()))
        # Process duplicates
        if process_duplicates_algo(sudoku_obj, possibility_dict, couple_list): return True
    return False


def process_duplicates_per_line(sudoku_obj, possibility_dict):
    for line_num in range(9):
        # Filter possibility per line
        couple_list = list(filter(lambda x: x[0] == line_num, possibility_dict.keys()))
        # Process duplicates
        if process_duplicates_algo(sudoku_obj, possibility_dict, couple_list): return True
    return False


def process_duplicates_per_row(sudoku_obj, possibility_dict):
    for row_num in range(9):
        # Filter possibility per line
        couple_list = list(filter(lambda x: x[1] == row_num, possibility_dict.keys()))
        # Process duplicates
        if process_duplicates_algo(sudoku_obj, possibility_dict, couple_list): return True
    return False


def process_duplicates_algo(sudoku_obj, possibility_dict, couple_list):
    ret_value = False

    # Filter couples with 2 num
    tmp_list = list(filter(lambda x: len(possibility_dict[x]) == 2, couple_list))

    # No duplicates found
    if len(tmp_list) < 2: return False

    # List tmp_couple per nums
    tmp_dict = dict()
    for tmp_couple in tmp_list:
        tmp_values = str(possibility_dict[tmp_couple])
        if not tmp_values in tmp_dict.keys():
            tmp_dict[tmp_values] = list()
        tmp_dict[tmp_values].append(tmp_couple)

    # Filter keep only duplicates
    duplicate_dict = dict()
    for tmp_values in tmp_dict.keys():
        if len(tmp_dict[tmp_values]) == 2:
            duplicate_dict[tmp_values] = tmp_dict[tmp_values]

    # For each duplicates
    for duplicate_values in duplicate_dict.keys():
        # Update couple_list
        couple_list = list(filter(lambda x: not x in duplicate_dict[duplicate_values], couple_list))

        # Filter duplicates values in other couples
        for tmp_couple in couple_list:
            for num_value in eval(duplicate_values):
                if num_value in possibility_dict[tmp_couple]:
                    ret_value = True  # Rescan need to process unique
                    possibility_dict[tmp_couple].remove(num_value)

        # Search unique num in couple list
        num_list = sudoku_obj.flatten_list(list(map(lambda x: possibility_dict[x], couple_list)))
        unique_num = list()
        for num_value in list(set(num_list)):
            if num_list.count(num_value) == 1:
                unique_num.append(num_value)

        # Apply unique num value in possibility_dict
        for tmp_couple in couple_list:
            # Apply unique num
            for unique_value in unique_num:
                if unique_value in possibility_dict[tmp_couple]:
                    ret_value = True
                    if sudoku_obj.get_value(tmp_couple) == 0:
                        # print("set_value "+str(tmp_couple)+" = "+str(unique_value))
                        sudoku_obj.set_value(unique_value, tmp_couple)

    return ret_value


def filter_duplicates(sudoku_obj, possibility_dict, couple_list):
    ret_value = False

    # List all num
    num_list = sudoku_obj.flatten_list(list(map(lambda x: possibility_dict[x], couple_list)))

    # Search each num declared twice
    twice_num = list()
    for num_value in list(set(num_list)):
        if num_list.count(num_value) == 2:
            twice_num.append(num_value)

    # Is two num find twice ?
    if len(twice_num) < 2:
        return

    # List all num couples duplicates possibility
    num_couples = list()
    for i, num_value in enumerate(twice_num):
        num_couples += list(product([num_value], twice_num[i + 1:]))

    # List duplicates couples possiblity
    eval_duplicate_couples = dict()
    for num_couple in num_couples:
        eval_duplicate_couples[num_couple] = list()
        for tmp_couple in couple_list:
            if num_couple[0] in possibility_dict[tmp_couple] and num_couple[1] in possibility_dict[tmp_couple]:
                eval_duplicate_couples[num_couple].append(tmp_couple)

    # Update duplicate list
    duplicate_values = list()
    duplicate_couples = list()
    for tmp_couple in eval_duplicate_couples.keys():
        if len(eval_duplicate_couples[tmp_couple]) > 1:
            ret_value = True
            duplicate_values = tmp_couple
            duplicate_couples = eval_duplicate_couples[tmp_couple]
            break  # Limit speed ?

    # Update possiblity_dict
    for tmp_couple in duplicate_couples:
        possibility_dict[tmp_couple] = list(duplicate_values)


def filter_duplicates_per_row(sudoku_obj, possibility_dict):
    for row_num in range(9):
        # Filter possibility per line
        couple_list = list(filter(lambda x: x[1] == row_num, possibility_dict.keys()))
        # Filter duplicate
        filter_duplicates(sudoku_obj, possibility_dict, couple_list)


def filter_duplicates_per_line(sudoku_obj, possibility_dict):
    for line_num in range(9):
        # Filter possibility per line
        couple_list = list(filter(lambda x: x[0] == line_num, possibility_dict.keys()))
        # Filter duplicate
        ret_value = filter_duplicates(sudoku_obj, possibility_dict, couple_list)


def filter_duplicates_per_block(sudoku_obj, possibility_dict):
    for block_num in range(9):
        # Filter possibility per block
        couple_list = list(filter(lambda x: block_num == int(x[0] / 3) * 3 + int(x[1] / 3), possibility_dict.keys()))
        # Filter duplicate
        ret_value = filter_duplicates(sudoku_obj, possibility_dict, couple_list)


def filter_combinations(sudoku_obj, possibility_dict):
    ret_value = False

    for block_num in range(9):
        current_block_tuple = ((block_num % 3), int(block_num / 3))

        # Parse each line
        line_list = list(map(lambda x: x + int(block_num / 3) * 3, list(range(3))))

        for line_num in line_list:
            # Keep possibility in line
            tmp_list = list(filter(lambda x: x[0] == line_num, possibility_dict.keys()))
            tmp_list = list(filter(lambda x: block_num != int(x[0] / 3) * 3 + int(x[1] / 3), tmp_list))

            # Get values
            tmp_values = list(map(lambda x: possibility_dict[x], tmp_list))
            tmp_values = list(set(sudoku_obj.flatten_list(tmp_values)))

            # Search value not present
            to_remove_values = list(range(1, 10))
            to_remove_values = list(filter(lambda x: not x in tmp_values, to_remove_values))

            # Filter num set
            to_remove_values = list(filter(lambda x: not x in sudoku_obj.get_line(line_num), to_remove_values))

            if len(to_remove_values) == 0: continue

            # Values not present should be removed in others line in the current block
            other_line_list = list(filter(lambda x: line_num != x, line_list))

            for other_line in other_line_list:
                # Filter couple to update
                update_list = list(filter(lambda x: x[0] == other_line, possibility_dict.keys()))
                update_list = list(filter(lambda x: block_num == int(x[0] / 3) * 3 + int(x[1] / 3), update_list))
                for tmp_couple in update_list:
                    tmp_len = len(possibility_dict[tmp_couple])
                    possibility_dict[tmp_couple] = list(
                        filter(lambda x: not x in to_remove_values, possibility_dict[tmp_couple]))
                    if tmp_len != len(possibility_dict[tmp_couple]):
                        ret_value = True

        # Parse each row
        row_list = list(map(lambda x: x + int(block_num % 3) * 3, list(range(3))))

        for row_num in row_list:
            # Keep possibility in line
            tmp_list = list(filter(lambda x: x[1] == row_num, possibility_dict.keys()))
            tmp_list = list(filter(lambda x: block_num != int(x[0] / 3) * 3 + int(x[1] / 3), tmp_list))

            # Get values
            tmp_values = list(map(lambda x: possibility_dict[x], tmp_list))
            tmp_values = list(set(sudoku_obj.flatten_list(tmp_values)))

            # Search value not present
            to_remove_values = list(range(1, 10))
            to_remove_values = list(filter(lambda x: not x in tmp_values, to_remove_values))

            # Filter num set
            to_remove_values = list(filter(lambda x: not x in sudoku_obj.get_row(row_num), to_remove_values))

            if len(to_remove_values) == 0: continue

            # Values not present should be removed in others line in the current block
            other_row_list = list(filter(lambda x: row_num != x, row_list))

            for other_row in other_row_list:
                # Filter couple to update
                update_list = list(filter(lambda x: x[1] == other_row, possibility_dict.keys()))
                update_list = list(filter(lambda x: block_num == int(x[0] / 3) * 3 + int(x[1] / 3), update_list))
                for tmp_couple in update_list:
                    tmp_len = len(possibility_dict[tmp_couple])
                    possibility_dict[tmp_couple] = list(
                        filter(lambda x: not x in to_remove_values, possibility_dict[tmp_couple]))
                    if tmp_len != len(possibility_dict[tmp_couple]):
                        ret_value = True

        # TODO in intersection take into account unique possibility

    return ret_value


################################################################
#
#  ######   #######  ##       ##     ## ######## ########  
# ##    ## ##     ## ##       ##     ## ##       ##     ## 
# ##       ##     ## ##       ##     ## ##       ##     ## 
#  ######  ##     ## ##       ##     ## ######   ########  
#       ## ##     ## ##        ##   ##  ##       ##   ##   
# ##    ## ##     ## ##         ## ##   ##       ##    ##  
#  ######   #######  ########    ###    ######## ##     ## 
#
################################################################
def get_num_per_couples(num_dict):
    # Create new dict
    ret_dict = dict()
    for num_value in num_dict.keys():
        for tmp_couple in num_dict[num_value]:
            if not tmp_couple in ret_dict.keys():
                ret_dict[tmp_couple] = list()
            ret_dict[tmp_couple].append(num_value)
    return ret_dict


def solver(sudoku_obj, iteration, verbose=True):
    # Scan each couples
    num_dict = dict()
    for num_value in range(1, 10):
        remain_couples = scan_num(sudoku_obj, num_value)
        num_dict[num_value] = remain_couples

    # For each couple get nums possibility
    possibility_dict = get_num_per_couples(num_dict)

    for i in range(iteration):
        if (verbose): print('--------------------------------')
        if (verbose): print('Iteration {}:'.format(i))
        rescan_cnt = 0

        # Update possiblity
        scan_update(sudoku_obj, possibility_dict)

        # Process unique
        rescan = process_unique(sudoku_obj, possibility_dict)
        if rescan:
            scan_update(sudoku_obj, possibility_dict)
            rescan_cnt += 1

        # Process duplicates
        rescan = process_duplicates(sudoku_obj, possibility_dict)
        if rescan:
            scan_update(sudoku_obj, possibility_dict)
            rescan_cnt += 1

        # Filter combinations
        # rescan = filter_combinations(sudoku_obj,possibility_dict)
        # if rescan:
        #     scan_update(sudoku_obj,possibility_dict)
        #     rescan_cnt += 1

        # No more idea
        if rescan_cnt == 0:
            break

    # Get remain values
    remain_values = sudoku_obj.get_remain_values()

    # Return status
    if list(remain_values.values()) == [0] * 9:
        # Solved
        return 0
    else:
        possibility_num = len(possibility_dict.keys())
        remain_num = sudoku_obj.flatten_list(sudoku_obj.matrix).count(0)

        # No solution found
        if possibility_num == remain_num:
            return 1
        # Error
        else:
            return -1


def sudoku_solver(sudoku_obj, iteration, verbose=False):
    # Copy Sudoku
    sudoku_copy = deepcopy(sudoku_obj)

    # Run solver
    ret_value = solver(sudoku_copy, iteration, verbose)

    # SOLVED
    if ret_value == 0:
        sudoku_obj.copy(sudoku_copy)
        # sudoku_obj.print_matrix()
        # print('----')
        return ret_value
    # ERROR
    elif ret_value == -1:
        return ret_value
    # NO SOLUTION FOUND
    elif ret_value == 1:
        # Scan each couples
        num_dict = dict()
        for num_value in range(1, 10):
            remain_couples = scan_num(sudoku_copy, num_value)
            num_dict[num_value] = remain_couples

        # List possiblity
        possibility_dict = get_num_per_couples(num_dict)
        possibility_len = list(map(lambda x: len(possibility_dict[x]), possibility_dict.keys()))

        if len(possibility_len) == 0:
            return -1

        # Test by possibility len (min to max) 
        for tmp_len in range(min(possibility_len), max(possibility_len) + 1):
            tmp_possibility = list(filter(lambda x: len(possibility_dict[x]) == tmp_len, possibility_dict.keys()))
            # Test each couples
            for tmp_couple in tmp_possibility:
                # Test each values
                for tmp_num in possibility_dict[tmp_couple]:
                    # Set value
                    sudoku_copy.set_value(tmp_num, tmp_couple)
                    # Run solver
                    ret_value = sudoku_solver(sudoku_copy, iteration, verbose)

                    if ret_value == 0:
                        sudoku_obj.copy(sudoku_copy)
                        return ret_value

                    # Erase value
                    sudoku_copy.set_value(0, tmp_couple)

                    if ret_value == -1:
                        continue
                    else:
                        return ret_value

        return ret_value


################################################################
#
# ##     ##    ###    #### ##    ##
# ###   ###   ## ##    ##  ###   ##
# #### ####  ##   ##   ##  ####  ##
# ## ### ## ##     ##  ##  ## ## ##
# ##     ## #########  ##  ##  ####
# ##     ## ##     ##  ##  ##   ###
# ##     ## ##     ## #### ##    ##
#
################################################################
def get_args():
    parser = argparse.ArgumentParser(
        description="Sudoku solver"
    )
    parser.add_argument('file')
    parser.add_argument('-i', '--iteration', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Load sudoku
    sudoku_obj = sudoku(args.file)

    print("Initial Matrix")
    sudoku_obj.print_matrix()
    sudoku_obj.print_remain_values()

    # Solve
    ret_value = sudoku_solver(sudoku_obj, args.iteration)

    if ret_value == 0:
        print('=> SOLVED')
    elif ret_value == 1:
        print('=> NO SOLUTION FOUND')
    elif ret_value == -1:
        print('=> ERROR')

    # Print output
    sudoku_obj.print_matrix()
    sudoku_obj.print_remain_values()

    print('--------------------------------')
    print("")
