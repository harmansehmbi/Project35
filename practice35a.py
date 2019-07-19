# KNN - K Nearest Neighbour

from sklearn.neighbors import KNeighborsClassifier


from sklearn.model_selection import train_test_split #Explore

# Representation of Data
trainingData = [ [100, 100],
         [150, 110],
         [180, 150],
         [200, 180],
         [800, 1000],
         [1000, 1200],
         [1200, 1300],
         [1500, 1500]
       ]
trainingLabels = [0, 0, 0, 0, 1, 1, 1, 1]

# X_train, X_test, Y_train, Y_test = train_test_split(dataSet, target, test_size=0.3)


testingData = [ [110, 105],
         [190, 110],
         [160, 100],
         [220, 170],
         [834, 1107],
         [1044, 1233],
         [1290, 1312],
         [1590, 1590]]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(trainingData, trainingLabels)

predictedLabels = model.predict(testingData)
print(predictedLabels)


