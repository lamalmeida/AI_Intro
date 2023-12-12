from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions

data = datasets.load_iris()
X = data['data'][:,:2]
y = data['target']
feature_names = data['feature_names'][:2]
target_names = data['target_names']

print(f'X has the shape {X.shape}')
print(f'y has the shape {y.shape}')
print(f'X has features: {feature_names}')
print(f'y has labels: {target_names}')

neigh = KNeighborsClassifier(n_neighbors = 15)
neigh.set_params()
neigh.fit(X, y)
train_score = neigh.score(X, y)
ax = plot_decision_regions(X, y, clf=neigh, legend=2)
plt.xlabel(f'{feature_names[0]}')
plt.ylabel(f'{feature_names[1]}')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, target_names)
plt.title('KNN on Iris')
plt.show()