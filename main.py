import collections
import math
import jieba

states = ['B', 'M', 'E', 'S']
init, trans, gen = [None] * 3
def fit(seqs):
    cinit = collections.defaultdict(lambda : 1)
    ctrans, cgen = [collections.defaultdict(lambda : collections.defaultdict(lambda : 1)) for _ in range(2)]
    for seq in seqs:
        observe = ''.join(seq)
        state = ''
        for term in seq:
            state += ('S' if len(term) == 1 else 'B' + 'M' * (len(term) - 2) + 'E')
        assert len(observe) == len(state)
        cinit[state[0]] += 1;
        for i in range(1, len(state)):
            ctrans[state[i - 1]][state[i]] += 1
        for i in range(len(state)):
            cgen[state[i]][observe[i]] += 1
    cnt = sum([cinit[k] for k in states])
    global init, trans, gen
    init, trans, gen = {}, {}, {}
    for state in states:
        init[state] = math.log(cinit[state] * 1.0 / cnt)
        ct = sum([ctrans[state][_] for _ in states])
        cg = sum([v for k, v in cgen[state].items()])
        trans[state], gen[state] = {}, collections.defaultdict(lambda : math.log(1.0 / cg))
        for nxt in states:
            trans[state][nxt] = math.log(ctrans[state][nxt] * 1.0 / ct)
        for k, v in cgen[state].items():
            gen[state][k] = math.log(v * 1.0 / cg)

def predict(seq):
    global init, trans, gen
    n = len(seq)
    path, dp = [[{} for l in range(n)] for _ in range(2)]
    for j in states:
        dp[0][j] = init[j] + gen[j][seq[0]]
        path[0][j] = j
    for i in range(1, n):
        for j in states:
            dp[i][j], path[i][j] = max([(dp[i - 1][k] + trans[k][j] + gen[j][seq[i]], path[i - 1][k] + j) for k in states])
    _, ob = max([(dp[n - 1][k], path[n - 1][k]) for k in states])
    ret, now = [], ''
    for i in range(n):
        if ob[i] == 'S':
            if len(now) > 0:
                ret.append(now)
                now = ''
            ret.append(seq[i])
        if ob[i] == 'M':
            now += seq[i]
        if ob[i] == 'E':
            now += seq[i]
            ret.append(now)
            now = ''
        if ob[i] == 'B':
            if len(now) > 0:
                ret.append(now)
                now = ''
            now += seq[i]
    if len(now) > 0:
        ret.append(now)
    return ret

def precision(test, ret):
    x, y = 0, 0
    for i in range(len(test)):
        for term in ret[i]:
            x += (1 if term in test[i] else 0)
            y += 1
    return x * 1.0 / y

def recall(test, ret):
    x, y = 0, 0
    for i in range(len(test)):
        for term in test[i]:
            x += (1 if term in ret[i] else 0)
            y += 1
    return x * 1.0 / y

if __name__ == '__main__':
    seqs = [[_.split('/')[0] for _ in line.split()[1 : ]] for line in open('data/renmin98.txt').readlines()]
    train = seqs[ : -3000]
    test = seqs[-3000 : ]
    fit(train)
    ret = [predict(''.join(_)) for _ in test]
    jb = [[v for v in jieba.cut(''.join(_))] for _ in test]
    
    Px, Py = precision(test, ret), precision(test, jb)
    Rx, Ry = recall(test, ret), recall(test, ret)
    print('HMM Precision:', Px)
    print('Jieba Precision:', Py)
    print('HMM Recall:', Rx)
    print('Jieba Recall:', Ry)
    print('HMM F1:', 2 * Px * Rx / (Px + Rx))
    print('Jieba F1:', 2 * Py * Ry / (Py + Ry))
 
