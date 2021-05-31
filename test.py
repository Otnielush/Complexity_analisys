
def compareTo(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    if str1 == str2:
        return 0
    elif len1 > len2:
        return 1
    else:
        return -1

# не понятно
def compare(query, text, i):
    M = len(query)
    j = 0
    while (i < N and j < M):
        if (query[j] != text[i]):
            return query[j] - text[i]
        i += 1
        j += 1

    if (i < N):
        return -1
    if (j < M):
        return +1
    return 0

class SuffixArrayJava6():
    def __init__(self, string):
        self.suffixes = list()
        self.n = len(string)
        for i in range(self.n):
            self.suffixes.append(string[i:])
        self.suffixes = sorted(self.suffixes)


    # size of string
    def length(self):
        return self.n


    # index of ith sorted suffix
    def index(self, i):
        return self.n - len(self.suffixes[i])


    # ith sorted suffix
    def select(self, i):
        return self.suffixes[i]


    # number of suffixes strictly less than query
    def rank(self, query):
        lo = 0
        hi = self.n - 1
        while (lo <= hi):
            mid = lo + (hi - lo) // 2
            cmp = compareTo(query, self.suffixes[mid])
            if (cmp < 0):
                hi = mid - 1
            elif (cmp > 0):
                lo = mid + 1
            else:
                return mid

        return lo


   # length of longest common prefix of s and t
    def _lcp(self, s: str, t: str):
        n = min(len(s), len(t))
        for i in range(n):
            if s[i] != t[i]:
                return i
        return n


    # longest common prefix of suffixes(i) and suffixes(i-1)
    # def lcp(self, i):
    #     return lcp(self.suffixes[i], self.suffixes[i-1])


    # longest common prefix of suffixes(i) and suffixes(j)
    def lcp(self, i, j):
        return self._lcp(self.suffixes[i], self.suffixes[j])




def max_shortest_substrings():
    text = 'ABRACADABRA!'

    suffix = SuffixArrayJava6(text)

    print("  i ind lcp rnk  select")
    print("---------------------------")

    mss = dict()

    for i in range(suffix.n):
        index = suffix.index(i)
        ith = "\"" + text[index: min(index + 50, len(text))] + "\""
        # String ith = suffix.select(i);
        rank = suffix.rank(suffix.select(i))
        if (i == 0):
            print("{:3d} {:3d} {:3s} {:3d}  {}".format( i, index, " -", rank, ith))
        else:
            lcp = suffix.lcp(i, i-1)
            mss[index] = lcp
            print("{:3d} {:3d} {:3d} {:3d}  {}".format( i, index, lcp, rank, ith))

    print(sorted(mss.items()))


max_shortest_substrings()

