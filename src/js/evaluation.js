/**
 * BloomSphere Evaluation Module
 * Executes shuffles, splits, and evaluates accuracy, precision, recall, F1, and Confusion Matrix.
 */

import { KNNClassifier } from './knn.js';
import { calculateStats, standardizeDataset } from './preprocessing.js';

/**
 * Creates a seeded random number generator (LCG).
 * @param {number} seed - Seed value.
 * @returns {Function} Generator function returning numbers in [0, 1).
 */
function createSeededRandom(seed) {
  let s = seed;
  return function() {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

/**
 * Fisher-Yates shuffle algorithm using a seeded random generator for reproducibility.
 * @param {Array} array - Original array.
 * @param {number} seed - Seed value.
 * @returns {Array} Shuffled copy of the array.
 */
export function shuffleArraySeeded(array, seed = 42) {
  const arr = [...array];
  const random = createSeededRandom(seed);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Fisher-Yates shuffle algorithm to randomize array order (kept for backward compatibility).
 * @param {Array} array - Original array.
 * @returns {Array} Shuffled copy of the array.
 */
export function shuffleArray(array) {
  return shuffleArraySeeded(array, Math.floor(Math.random() * 1000));
}

/**
 * Evaluates the model performance using a train-test split.
 * @param {Array} rawDataset - Unprocessed Iris dataset.
 * @param {number} k - Number of nearest neighbors (default 5).
 * @param {number} trainRatio - Partition ratio for training (default 0.8).
 * @returns {Object} Evaluation results containing metrics and the confusion matrix.
 */
export function evaluateModel(rawDataset, k = 5, trainRatio = 0.8) {
  // 1. Shuffle raw dataset using a seeded random function to make it deterministic (seed = 6)
  const shuffled = shuffleArraySeeded(rawDataset, 6);
  
  // 2. Split train/test
  const splitIndex = Math.floor(shuffled.length * trainRatio);
  const rawTrainSet = shuffled.slice(0, splitIndex);
  const rawTestSet = shuffled.slice(splitIndex);

  // 3. Compute scaling statistics *only* on the training set (proper Z-score scaling, preventing data leakage!)
  const trainStats = calculateStats(rawTrainSet);
  
  // 4. Standardize both sets using trainStats
  const trainSet = standardizeDataset(rawTrainSet, trainStats);
  const testSet = standardizeDataset(rawTestSet, trainStats);

  // 5. Initialize KNN classifier
  const knn = new KNNClassifier(k);
  knn.setTrainingData(trainSet);

  // 6. Initialize Confusion Matrix
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

  // 7. Evaluate on Test Set
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

  // 8. Calculate accuracy
  const totalTest = testSet.length;
  const accuracy = totalTest > 0 ? correctCount / totalTest : 0;

  // 9. Calculate Precision, Recall, F1 for each class
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

  // 10. Compute Macro-averaged metrics
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
