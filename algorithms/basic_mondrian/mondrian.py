# -*- coding: utf-8 -*-

"""
main module of basic Mondrian
"""

import pdb
import time
from functools import cmp_to_key

from tqdm import tqdm
from .models.numrange import NumRange
from .utils.utility import cmp_str
from collections import defaultdict

__DEBUG = False
QI_LEN = 8
SA_INDEX = []
GL_K = 0
RESULT = []
ATT_TREES = []
QI_RANGE = []
IS_CAT = []
GL_L = 0

SUM_RES = 0
TOT_RES = 1


class Partition(object):

    """Class for Group, which is used to keep records
    Store tree node in instances.
    self.member: records in group
    self.width: width of this partition on each domain. For categoric attribute, it equal
    the number of leaf node, for numeric attribute, it equal to number range
    self.middle: save the generalization result of this partition
    self.allow: 0 donate that not allow to split, 1 donate can be split
    """

    def __init__(self, data, width, middle):
        """
        initialize with data, width and middle
        """
        self.member = list(data)
        self.width = list(width)
        self.middle = list(middle)
        self.allow = [1] * QI_LEN

    def __len__(self):
        """
        return the number of records in partition
        """
        return len(self.member)


def get_normalized_width(partition, index):
    """
    return Normalized width of partition
    similar to NCP
    """
    if not IS_CAT[index]:
        low = partition.width[index][0]
        high = partition.width[index][1]
        width = float(ATT_TREES[index].sort_value[high]) - float(
            ATT_TREES[index].sort_value[low]
        )
    else:
        width = partition.width[index]
    if not QI_RANGE[index]:
        # # print("DBG::", "QI_RANGE at", index, "collapsed!!!")
        return width * 1.0 / (QI_RANGE[index] + 0.001)
    else:
        return width * 1.0 / QI_RANGE[index]


def choose_dimension(partition):
    """
    chooss dim with largest normlized Width
    return dim index.
    """
    max_width = -1
    max_dim = -1
    for i in range(QI_LEN):
        if not partition.allow[i]:
            continue
        normWidth = get_normalized_width(partition, i)
        normWidth *= QI_WEIGHT[i]
        if normWidth > max_width:
            max_width = normWidth
            max_dim = i
    if max_width > 1:
        print("Error: max_width > 1")
        pdb.set_trace()
    if max_dim == -1:
        print("cannot find the max dim")
        pdb.set_trace()
    return max_dim


def frequency_set(partition, dim):
    """
    get the frequency_set of partition on dim
    return dict{key: str values, values: count}
    """
    frequency = {}
    for record in partition.member:
        try:
            frequency[record[dim]] += 1
        except KeyError:
            frequency[record[dim]] = 1
    return frequency


def find_median(partition, dim):
    """
    find the middle of the partition
    return splitVal
    """
    frequency = frequency_set(partition, dim)
    splitVal = ""
    value_list = list(frequency)
    value_list.sort(key=cmp_to_key(cmp_str))
    total = sum(frequency.values())
    middle = total / 2

    # value_list = frequency.keys()
    # value_list.sort(cmp=cmp_str)
    # total = sum(frequency.values())
    # middle = total / 2

    if GL_L:
        if middle < GL_L or len(value_list) <= 1:  # CONST L
            return ("", "", value_list[0], value_list[-1])
    elif GL_K:
        if middle < GL_K or len(value_list) <= 1:  # CONST K
            return ("", "", value_list[0], value_list[-1])

    index = 0
    split_index = 0

    for i, t in enumerate(value_list):
        index += frequency[t]
        if index >= middle:
            splitVal = t
            split_index = i
            break
    else:
        print("Error: cannot find splitVal")

    try:
        nextVal = value_list[split_index + 1]
    except IndexError:
        nextVal = splitVal

    return (splitVal, nextVal, value_list[0], value_list[-1])


def split_numerical_value(numeric_value, splitVal):
    """
    split numeric value on splitVal
    return sub ranges
    """
    split_num = numeric_value.split("~")
    if len(split_num) <= 1:
        return split_num[0], split_num[0]
    else:
        low = split_num[0]
        high = split_num[1]
        # Fix 2,2 problem
        if low == splitVal:
            lvalue = low
        else:
            lvalue = low + "~" + splitVal
        if high == splitVal:
            rvalue = high
        else:
            rvalue = splitVal + "~" + high
        return lvalue, rvalue


def split_numerical(partition, dim, pwidth, pmiddle):
    """
    strict split numeric attribute by finding a median,
    lhs = [low, means], rhs = (mean, high]
    """
    sub_partitions = []
    # numeric attributes
    (splitVal, nextVal, low, high) = find_median(partition, dim)
    p_low = ATT_TREES[dim].dict[low]
    p_high = ATT_TREES[dim].dict[high]
    # update middle
    if low == high:
        pmiddle[dim] = low
    else:
        pmiddle[dim] = low + "~" + high
    pwidth[dim] = (p_low, p_high)
    if splitVal == "" or splitVal == nextVal:
        # update middle
        return []
    middle_pos = ATT_TREES[dim].dict[splitVal]
    lmiddle = pmiddle[:]
    rmiddle = pmiddle[:]
    lmiddle[dim], rmiddle[dim] = split_numerical_value(pmiddle[dim], splitVal)
    lhs = []
    rhs = []
    for temp in partition.member:
        pos = ATT_TREES[dim].dict[temp[dim]]
        if pos <= middle_pos:
            # lhs = [low, means]
            lhs.append(temp)
        else:
            # rhs = (mean, high]
            rhs.append(temp)
    lwidth = pwidth[:]
    rwidth = pwidth[:]
    lwidth[dim] = (pwidth[dim][0], middle_pos)
    rwidth[dim] = (ATT_TREES[dim].dict[nextVal], pwidth[dim][1])
    if GL_L:
        if check_L_diversity(lhs) is False or check_L_diversity(rhs) is False:
            return []
    sub_partitions.append(Partition(lhs, lwidth, lmiddle))
    sub_partitions.append(Partition(rhs, rwidth, rmiddle))
    return sub_partitions


def split_categorical(partition, dim, pwidth, pmiddle):
    """
    split categorical attribute using generalization hierarchy
    """
    # na_count = 0
    node_to_split = ATT_TREES[dim][partition.middle[dim]]
    children = [t for t in node_to_split.child]

    if not children:
        return []  # split is not necessary

    row_groups = [[] for _ in range(len(children))] 

    for row in partition.member:
        val = row[dim]
        for i, node in enumerate(children):
            try:
                node.cover[val]
            except KeyError:
                continue
            else:
                row_groups[i].append(row)
                break
        else:
            print(f"`{val}`has not been found in current VGH")
            pass
            # if val == "":
            #     na_count += 1
            #     for j in range(len(children)):
            #         na_groups[i].append(row)

    splittable = True

    for i, group in enumerate(row_groups):
        if not group:
            continue

        if GL_L:
            if not check_L_diversity(group):
                splittable = False
                break
        elif GL_K:
            if len(group) < GL_K:
                splittable = False
                break

    if splittable:
        sub_partitions = list()
        for i, group in enumerate(row_groups):
            if not group:
                continue

            child = children[i]
            width_tmp = pwidth[:]
            middle_tmp = pmiddle[:]

            width_tmp[dim] = len(child)
            middle_tmp[dim] = child.value

            sub_partitions.append(Partition(group, width_tmp, middle_tmp))
        return sub_partitions
    else:
        return []


def split_partition(partition, dim):
    """
    split partition and distribute records to different sub-partitions
    """
    pwidth = partition.width
    pmiddle = partition.middle
    if not IS_CAT[dim]:
        return split_numerical(partition, dim, pwidth, pmiddle)
    else:
        return split_categorical(partition, dim, pwidth, pmiddle)


def anonymize(partition, level=0):
    """
    Main procedure of Half_Partition.
    recursively partition groups until not allowable.
    """
    global SUM_RES, TOT_RES
    is_splittable = check_splitable(partition)
    if not is_splittable:
        RESULT.append(partition)

        SUM_RES += len(partition)
        pcent = SUM_RES / TOT_RES
        print("DBG::", f"{pcent:.2%}")
        if pcent > 1 and level > 11:
            pass

        return
    # Choose dim
    dim = choose_dimension(partition)
    # print("DBG::", "--" * level, "+", ">", dim)
    if dim == -1:
        print("Error: dim=-1")
        pdb.set_trace()

    sub_partitions = split_partition(partition, dim)
    # print("DBG::", "-" * level, "+", dim, "<", len(sub_partitions))

    if not sub_partitions:
        partition.allow[dim] = 0
        anonymize(partition, level=level)
    else:
        for sub_p in sub_partitions:
            anonymize(sub_p, level=level + 1)


def check_splitable(partition):
    """
    Check if the partition can be further splited while satisfying k-anonymity.
    """
    return True if sum(partition.allow) else False


def init(att_trees, data, QI_num, SA_num, k=None, L=None, QI_weight=None):
    """
    reset all global variables
    """
    global GL_K, RESULT, QI_LEN, ATT_TREES, QI_RANGE, IS_CAT, SA_INDEX, GL_L, QI_WEIGHT
    ATT_TREES = att_trees
    for t in att_trees:
        if isinstance(t, NumRange):
            IS_CAT.append(False)
        else:
            IS_CAT.append(True)

    if QI_num <= 0:
        QI_LEN = len(data[0]) - 1  # if there's no QI, QI = ID at 1st column
    else:
        QI_LEN = QI_num

    SA_INDEX = SA_num
    QI_WEIGHT = QI_weight if QI_weight is not None else [1] * QI_LEN
    RESULT = []
    QI_RANGE = []
    if k is not None:
        GL_K = k
    else:
        GL_K = 0
    if L is not None:
        GL_L = L
    else:
        GL_L = 0


_ESC_CHARS = ["", "*"]


def check_L_diversity(partition, T_closeness=False):
    """check if partition satisfy l-diversity
    return True if satisfy, False if not.
    """

    if len(partition) < GL_L:
        return False

    records_set = partition.member if isinstance(partition, Partition) else partition

    for idx in SA_INDEX:
        if not T_closeness:
            val_set = set(r[idx] for r in records_set if r[idx] not in _ESC_CHARS)
            if 0 < len(val_set) < GL_L:
                return False
        else:
            sa_dict = defaultdict(int)
            for record in records_set:
                sa_value = record[idx]
                if sa_value not in ["", "*"]:  # possible nan values
                    sa_dict[sa_value] += 1

            if 0 < len(sa_dict) < GL_L:
                return False

            for k, sa_freq in sa_dict.items():
                # if any SA value appear more than |T|/l,
                # the partition does not satisfy l-diversity
                if sa_freq > len(records_set) / GL_L:
                    return False
    return True


def mondrian(att_trees, data, k, QI_num, SA_num):
    """
    basic Mondrian for k-anonymity.
    This fuction support both numeric values and categoric values.
    For numeric values, each iterator is a mean split.
    For categoric values, each iterator is a split on GH.
    The final result is returned in 2-dimensional list.
    """
    init(att_trees, data, QI_num, SA_num, k=k)
    result = []
    middle = []
    wtemp = []
    for i in tqdm(range(QI_LEN)):
        if IS_CAT[i] is False:
            QI_RANGE.append(ATT_TREES[i].range)
            wtemp.append((0, len(ATT_TREES[i].sort_value) - 1))
            middle.append(ATT_TREES[i].value)
        else:
            QI_RANGE.append(len(ATT_TREES[i]["*"]))
            wtemp.append(len(ATT_TREES[i]["*"]))
            middle.append("*")
    whole_partition = Partition(data, wtemp, middle)
    start_time = time.time()
    anonymize(whole_partition)
    rtime = float(time.time() - start_time)
    #
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ERRORE QUI - NON PRENDE UID e ALTRI DATI
    for partition in RESULT:
        temp = partition.middle
        for i in range(len(partition)):
            temp_for_SA = []
            for s in range(
                len(partition.member[i]) - len(SA_INDEX), len(partition.member[i])
            ):
                temp_for_SA = temp_for_SA + [partition.member[i][s]]
            result.append(temp + temp_for_SA)
    return (result, rtime)


def mondrian_l_diversity(att_trees, data, L, QI_num, SA_num, QI_weight=None):
    """
    Mondrian for l-diversity.
    This fuction support both numeric values and categoric values.
    For numeric values, each iterator is a mean split.
    For categoric values, each iterator is a split on GH.
    The final result is returned in 2-dimensional list.
    """
    global TOT_RES
    init(att_trees, data, QI_num, SA_num, L=L, QI_weight=QI_weight)
    middle = []
    result = []
    wtemp = []
    for i in range(QI_LEN):
        if not IS_CAT[i]:
            # print("DBG::", ATT_TREES[i].range, "at", i)
            QI_RANGE.append(ATT_TREES[i].range)
            wtemp.append((0, len(ATT_TREES[i].sort_value) - 1))
            middle.append(ATT_TREES[i].value)
        else:
            # print("DBG::", len(ATT_TREES[i]['*']), "at", i)
            QI_RANGE.append(len(ATT_TREES[i]["*"]))
            wtemp.append(len(ATT_TREES[i]["*"]))
            middle.append("*")
    whole_partition = Partition(data, wtemp, middle)
    TOT_RES = len(whole_partition)
    start_time = time.time()
    anonymize(whole_partition)
    rtime = float(time.time() - start_time)

    dp = 0.0
    for partition in RESULT:
        dp += len(partition) ** 2
        temp = partition.middle
        for i in range(len(partition)):
            temp_for_SA = []
            for s in range(len(temp), len(partition.member[i])):
                temp_for_SA = temp_for_SA + [partition.member[i][s]]
            result.append(temp + temp_for_SA)

    return (result, rtime)
