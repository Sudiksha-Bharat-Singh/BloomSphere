/**
 * BloomSphere KNN Classifier Module
 * Contains scratch implementation of K-Nearest Neighbors algorithm.
 */

export class KNNClassifier {
  /**
   * Initializes the KNN Classifier.
   * @param {number} k - Number of nearest neighbors (default 5).
   */
  constructor(k = 5) {
    this.k = k;
    this.trainingData = [];
  }

  /**
   * Sets the reference training dataset.
   * @param {Array} data - Array of standardized feature vectors.
   */
  setTrainingData(data) {
    this.trainingData = data;
  }

  /**
   * Calculates Euclidean distance between two 4-dimensional vectors.
   * @param {Object} p1 - First point with sepal/petal dimensions.
   * @param {Object} p2 - Second point with sepal/petal dimensions.
   * @returns {number} Euclidean distance.
   */
  euclideanDistance(p1, p2) {
    return Math.sqrt(
      Math.pow(Number(p1.sepalLength) - Number(p2.sepalLength), 2) +
      Math.pow(Number(p1.sepalWidth) - Number(p2.sepalWidth), 2) +
      Math.pow(Number(p1.petalLength) - Number(p2.petalLength), 2) +
      Math.pow(Number(p1.petalWidth) - Number(p2.petalWidth), 2)
    );
  }

  /**
   * Predicts the species class label for a custom query sample.
   * @param {Object} queryPoint - Raw feature dimensions of query.
   * @returns {Object} Prediction result including species class name and neighbor details.
   */
  predict(queryPoint) {
    if (this.trainingData.length === 0) {
      throw new Error('Training data is not loaded. Call setTrainingData() first.');
    }

    // 1. Calculate distance from queryPoint to all training samples
    const distances = this.trainingData.map(sample => {
      const dist = this.euclideanDistance(queryPoint, sample);
      return {
        distance: dist,
        species: sample.species,
        sample: sample
      };
    });

    // 2. Sort distances in ascending order
    distances.sort((a, b) => a.distance - b.distance);

    // 3. Select the k nearest neighbors
    const nearestNeighbors = distances.slice(0, this.k);

    // 4. Perform majority voting
    const votes = {};
    nearestNeighbors.forEach(neighbor => {
      const sp = neighbor.species.toLowerCase();
      votes[sp] = (votes[sp] || 0) + 1;
    });

    // 5. Find winner
    let predictedSpecies = null;
    let maxVotes = -1;

    Object.keys(votes).forEach(sp => {
      if (votes[sp] > maxVotes) {
        maxVotes = votes[sp];
        predictedSpecies = sp;
      } else if (votes[sp] === maxVotes) {
        // Tie breaker: Select the class that has the absolute closest single neighbor
        const firstOfCurrent = nearestNeighbors.find(n => n.species.toLowerCase() === sp);
        const firstOfWinner = nearestNeighbors.find(n => n.species.toLowerCase() === predictedSpecies);
        
        if (firstOfCurrent && firstOfWinner && firstOfCurrent.distance < firstOfWinner.distance) {
          predictedSpecies = sp;
        }
      }
    });

    return {
      species: predictedSpecies, // lowercase setosa, versicolor, virginica
      neighbors: nearestNeighbors,
      votes: votes
    };
  }
}
