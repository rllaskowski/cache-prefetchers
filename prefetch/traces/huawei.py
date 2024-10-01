import operator
import os
import sys


def parse_huawei_trace_file(filename):
    res = []
    header = None
    with open(filename) as f:
        for line in f:
            s = [x.rstrip() for x in line.rstrip().split('|')]
            if header is None:
                header = s
                i_op = header.index('tp')
                i_obj = header.index('objId')
                i_lba = header.index('objLba')
                i_t = header.index('last time nano')
                i_len = header.index('length')
                cols_count = len(header)
                if ValueError in [i_op, i_obj, i_lba, i_t, i_len]:
                    print('Error parsing header', filename, file=sys.stderr)
                    return None
            elif cols_count <= len(s):  # some traces are cut in the middle, we have to protect against it
                op = s[i_op]
                if op in ['READ', 'WRITE']:
                    obj = int(s[i_obj], 0)
                    lba = int(s[i_lba])
                    length = int(s[i_len])
                    t = int(s[i_t])
                    res.append((op, obj, lba, length, t))
    return sorted(res, key=operator.itemgetter(4))


def parse_huawei_traces_dir(path):
    res = []
    for root, dirs, files in os.walk(path):
        for name in files:
            v = parse_huawei_trace_file(os.path.join(root, name))
            if v is None:
                return None
            res += v
    return sorted(res, key=operator.itemgetter(4))


def parse_huawei_traces(path):
    """
    :param path: path to traces
    :type path: str
    :return: list of tuples (op, lbs, time) where
        op is 'READ' or 'WRITE',
        lbs is page number,
        time is a numeric timestamp
    :rtype: list[(str, int, int)]
    """
    if os.path.isfile(path):
        return parse_huawei_trace_file(path)
    elif os.path.isdir(path):
        return parse_huawei_traces_dir(path)
    else:
        print('Cannot read', path, file=sys.stderr)
        return None


def page_seq(traces, cluster_size, only_reads=True):
    seq = []
    for (op, obj, lba, length, time) in traces:
        if only_reads and op == 'READ':
            continue
        length *= 512
        start_page = lba // cluster_size
        end_page = (lba + length - 1) // cluster_size
        for page in range(start_page, end_page + 1):
            seq.append(256 * page + obj)
    return seq


def dump_trace_to_file(path, file):
    f = open(file, 'w')
    for page in page_seq(parse_huawei_traces(path)):
        f.write("%d\n" % page)
    f.close()
