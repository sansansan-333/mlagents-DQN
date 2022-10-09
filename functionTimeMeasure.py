import time

### Measures elapsed time of each function or section between start and end. ###
class FunctionTimeMeasure: 
    def __init__(self):
        self._sum = {}
        self._start = {}
        self._total_start = None

    def start(self, name):
        if self._total_start == None:
            self._total_start = time.time()

        if not name in self._sum.keys():
            self._sum[name] = 0
        self._start[name] = time.time()

    def end(self, name):
        if name in self._start.keys():
            self._sum[name] += time.time() - self._start[name]

    def get(self, name):
        return self._sum[name]

    def print(self):
        if self._total_start == None:
            return

        total = time.time() - self._total_start
        sorted_sum = {k: v for k, v in sorted(self._sum.items(), key=lambda item: item[1], reverse=True)}
        print(f'Total time elapsed while measuring: {total}')
        for sum in sorted_sum.items():
            print(f'    {sum[0]}: {sum[1]} ({sum[1] / total * 100} %)')