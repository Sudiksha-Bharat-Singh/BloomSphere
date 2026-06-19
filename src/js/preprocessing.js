/**
 * BloomSphere Preprocessing Module
 * Handles standardization / feature scaling (Z-score normalization).
 */

/**
 * Calculates mean and standard deviation for each of the 4 numerical features.
 * @param {Array} dataset - Array of sample objects.
 * @returns {Object} Statistics containing mean and stdDev for each feature.
 */
export function calculateStats(dataset) {
  const features = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'];
  const stats = {};

  features.forEach(feat => {
    const values = dataset.map(item => Number(item[feat]));
    const sum = values.reduce((acc, val) => acc + val, 0);
    const mean = sum / values.length;
    
    const sqDiffSum = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
    const variance = sqDiffSum / values.length;
    const stdDev = Math.sqrt(variance);

    // Prevent division by zero if stdDev is 0
    stats[feat] = {
      mean: mean,
      stdDev: stdDev || 1
    };
  });

  return stats;
}

/**
 * Standardizes a dataset array using pre-computed feature statistics.
 * @param {Array} dataset - Array of sample objects.
 * @param {Object} stats - Calculated mean/stdDev statistics.
 * @returns {Array} New array containing standardized features.
 */
export function standardizeDataset(dataset, stats) {
  return dataset.map(item => {
    const standardizedItem = { ...item };
    ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'].forEach(feat => {
      standardizedItem[feat] = (Number(item[feat]) - stats[feat].mean) / stats[feat].stdDev;
    });
    return standardizedItem;
  });
}

/**
 * Standardizes a single query sample coordinates.
 * @param {Object} sample - Single measurement query point.
 * @param {Object} stats - Pre-calculated dataset statistics.
 * @returns {Object} Standardized sample copy.
 */
export function standardizeSample(sample, stats) {
  const standardizedSample = { ...sample };
  ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'].forEach(feat => {
    standardizedSample[feat] = (Number(sample[feat]) - stats[feat].mean) / stats[feat].stdDev;
  });
  return standardizedSample;
}
