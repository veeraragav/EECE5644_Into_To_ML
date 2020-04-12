import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
## ======================= Loading Dtrain and Dtest ===================== ##
csv_file = 'csv/Dtrain_p2.csv'
xy_train = np.loadtxt(csv_file, delimiter=',', dtype=np.float32)
x_train = xy_train[:, 1:]
y_train = np.squeeze(xy_train[:, [0]])
y_train = y_train - 1

csv_file = 'csv/Dtest_p2.csv'
xy_test = np.loadtxt(csv_file, delimiter=',', dtype=np.float32)
x_test = xy_test[:, 1:]
y_test = np.squeeze(xy_test[:, [0]])
y_test = y_test - 1

## ==================== Helper class =========================== ##
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


## =========================== K-fold ============================== ##
rbf_svc = svm.SVC(kernel='rbf')
#C_range = [0.01, 0.1, 1, 10, 100]
#gamma_range = [0.01, 0.1, 1, 10, 100]
#C_range = np.logspace(-1, 9, 11)
#gamma_range = np.logspace(-2, 3, 13)
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = KFold(n_splits=10)
grid = GridSearchCV(rbf_svc, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=10)
grid.fit(x_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

C_best = grid.best_params_['C']
sigma_best = grid.best_params_['gamma']

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('K-fold CrossValidation accuracy')
#plt.show()

## =============================== Training SVM with above selected hyperparameters and Dtrain ======================== ##
rbf_svc_best = svm.SVC(kernel='rbf', C=C_best, gamma=sigma_best)
rbf_svc_best.fit(x_train, y_train)

## =============================== Classification of Dtest ========================================= ##
predictions = rbf_svc_best.predict(x_test)
c0_classified_c0 = []
c1_classified_c1 = []
c1_classified_c0 = []
c0_classified_c1 = []
for i, class_predicted in enumerate(predictions):
    if class_predicted == y_test[i]:
        if class_predicted == 0.0:
            c0_classified_c0.append(x_test[i])
        elif class_predicted == 1.0:
            c1_classified_c1.append(x_test[i])
    else:
        if class_predicted == 0.0:
            c1_classified_c0.append(x_test[i])
        elif class_predicted == 1.0:
            c0_classified_c1.append(x_test[i])

p_correct = (len(c0_classified_c0) + len(c1_classified_c1)) / len(x_test)

print('c0_classified_as_c0:', len(c0_classified_c0))
print('c0_classified_as_c1:', len(c0_classified_c1))
print('c1_classified_as_c0:', len(c1_classified_c0))
print('c1_classified_as_c1:', len(c1_classified_c1))

print('P(correct): ', p_correct)

plt.figure()
plt.scatter([col[0] for col in c0_classified_c0], [col[1] for col in c0_classified_c0])
plt.scatter([col[0] for col in c1_classified_c1], [col[1] for col in c1_classified_c1])
plt.scatter([col[0] for col in c0_classified_c1], [col[1] for col in c0_classified_c1])
plt.scatter([col[0] for col in c1_classified_c0], [col[1] for col in c1_classified_c0])
plt.legend(['c0_classified_as_c0', ' c1_classified_as_c1', 'c0_classified_as_c1', 'c1_classified_as_c0'])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Classification of Dtest')
plt.show()



