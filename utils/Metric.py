from math import sqrt

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val   = 0
		self.avg   = 0
		self.sum   = 0
		self.count = 0

	def update(self, val, n=1):
		self.val   = val
		self.sum   += val * n
		self.count += n
		self.avg   = self.sum / self.count

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred    = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return

def Performance(predict, y_test):

    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    tp_idx = []
    tn_idx = []
    fp_idx = []
    fn_idx = []

    for i in range(len(y_test)):
        if (y_test[i] == 1):
            if (predict[i] == 1):
                tp += 1
                tp_idx.append(i)
            else:
                fn += 1
                fn_idx.append(i)
        if (y_test[i] == 0):
            if (predict[i] == 1):
                fp += 1
                fp_idx.append(i)
            else:
                tn += 1
                tn_idx.append(i)

    tpr, tnr, fpr, acc, mcc, fdr, f1 = -1, -1, -1, -1, -1, -1, -1
    try:
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        fdr = fp / (fp + tp)
        f1 = 2 * tp / (2 * tp + fp + fn)
    except Exception:
        pass
    return [tp, tn, fp, fn, tpr, tnr, fpr, acc, mcc, fdr, f1]
    #return [tp, tn, fp, fn, tpr, tnr, fpr, acc, mcc, fdr, f1, tp_idx, tn_idx, fp_idx, fn_idx]
