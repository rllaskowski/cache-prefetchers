

def read_trace():
    trace = []
    with open('pinatrace.out', 'r') as f:
        lines = f.readlines()

    for line in lines:
        if "#eof" in line:
            break
        inf = line.split()
        inf[0] = inf[0][:-1]

        trace.append((int(inf[0], 0), inf[1], int(inf[2], 0)))

    return trace
