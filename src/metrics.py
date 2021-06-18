from sklearn.metrics import accuracy_score, average_precision_score
from scipy.stats import spearmanr

def accuracy(y_true, y_pred):
	return accuracy_score(y_true, y_pred)

def spearman(y_true_score, y_pred_score):
	corr, _ = spearmanr(y_true_score, y_pred_score)
	return corr
