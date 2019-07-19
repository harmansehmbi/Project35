from sklearn.neighbors import KNeighborsClassifier

trainingData = [ [100, 100],
         [150, 110],
         [180, 150],
         [200, 180],
         [800, 1000],
         [1000, 1200],
         [1200, 1300],
         [1500, 1500],
       ]
trainingLabels = [0, 0, 0, 0, 1, 1, 1, 1]

# Model Creation
model = KNeighborsClassifier(n_neighbors=3)

# Model Training
model.fit(trainingData, trainingLabels)

sampleInput = [147, 215]
predictedClass = model.predict([sampleInput])
print(">> Predicted Class is:",predictedClass)
