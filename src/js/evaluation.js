/**
 * BloomSphere Evaluation Module
 * Executes shuffles, splits, and evaluates accuracy, precision, recall, F1, and Confusion Matrix.
 */

import { KNNClassifier } from './knn.js';

/**
 * Fisher-Yates shuffle algorithm to randomize array order.
 * @param {Array} array - Original array.
 * @returns {Array} Shuffled copy of the array.
 */
export function shuffleArray(array) {
  const arr = [...array];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Evaluates the model performance using a train-test split.
 * @param {Array} standardizedDataset - Preprocessed (scaled) Iris dataset.
 * @param {number} k - Number of nearest neighbors (default 5).
 * @param {number} trainRatio - Partition ratio for training (default 0.8).
 * @returns {Object} Evaluation results containing metrics and the confusion matrix.
 */
export function evaluateModel(standardizedDataset, k = 5, trainRatio = 0.8) {
  // 1. Shuffle
  const shuffled = shuffleArray(standardizedDataset);
  
  // 2. Split train/test
  const splitIndex = Math.floor(shuffled.length * trainRatio);
  const trainSet = shuffled.slice(0, splitIndex);
  const testSet = shuffled.slice(splitIndex);

  // 3. Initialize KNN classifier
  const knn = new KNNClassifier(k);
  knn.setTrainingData(trainSet);

  // 4. Initialize Confusion Matrix
  // Indexes: Setosa = 0, Versicolor = 1, Virginica = 2
  const classMap = {
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
  };

  const confusionMatrix = [
    [0, 0, 0], // row 0: Actual Setosa
    [0, 0, 0], // row 1: Actual Versicolor
    [0, 0, 0]  // row 2: Actual Virginica
  ];

  let correctCount = 0;

  // 5. Evaluate on Test Set
  testSet.forEach(sample => {
    const actualLabel = sample.species.toLowerCase();
    const actualIdx = classMap[actualLabel];
    if (actualIdx === undefined) return; // skip if invalid record

    const query = {
      sepalLength: sample.sepalLength,
      sepalWidth: sample.sepalWidth,
      petalLength: sample.petalLength,
      petalWidth: sample.petalWidth
    };

    const prediction = knn.predict(query);
    const predLabel = prediction.species.toLowerCase();
    const predIdx = classMap[predLabel];

    if (actualIdx === predIdx) {
      correctCount++;
    }

    confusionMatrix[actualIdx][predIdx]++;
  });

  // 6. Calculate accuracy
  const totalTest = testSet.length;
  const accuracy = totalTest > 0 ? correctCount / totalTest : 0;

  // 7. Calculate Precision, Recall, F1 for each class
  const classMetrics = [0, 1, 2].map(i => {
    const tp = confusionMatrix[i][i];

    // Column sum = True Positives + False Positives (predicted as i)
    let colSum = 0;
    for (let row = 0; row < 3; row++) {
      colSum += confusionMatrix[row][i];
    }
    const fp = colSum - tp;

    // Row sum = True Positives + False Negatives (actual class i)
    let rowSum = 0;
    for (let col = 0; col < 3; col++) {
      rowSum += confusionMatrix[i][col];
    }
    const fn = rowSum - tp;

    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

    return { precision, recall, f1 };
  });

  // 8. Compute Macro-averaged metrics
  const macroPrecision = classMetrics.reduce((sum, m) => sum + m.precision, 0) / 3;
  const macroRecall = classMetrics.reduce((sum, m) => sum + m.recall, 0) / 3;
  const macroF1 = classMetrics.reduce((sum, m) => sum + m.f1, 0) / 3;

  return {
    accuracy,
    precision: macroPrecision,
    recall: macroRecall,
    f1Score: macroF1,
    confusionMatrix,
    classMetrics
  };
}
