/**
 * BloomSphere Main Script
 * Manages central application state, initializations, and prediction events.
 */

import { irisDataset } from './dataset.js';
import { calculateStats, standardizeDataset, standardizeSample } from './preprocessing.js';
import { KNNClassifier } from './knn.js';
import { evaluateModel } from './evaluation.js';
import { initSliders, renderPredictionResult, renderMetricsDashboard, togglePredictionView } from './ui.js';

// 1. Central State Object
const state = {
  rawDataset: irisDataset,
  stats: null,               // Mean and stdDev statistics
  standardizedDataset: null, // Full dataset scaled
  kValue: 5,                 // Default k-value
  splitRatio: 0.8,           // 80% Train, 20% Test
  evaluationResults: null,   // Metrics (accuracy, precision, recall, f1, confusion matrix)
  predictorModel: null       // KNN model fitted on full standardized data for playground
};

// Expose state globally for verification
window.state = state;

document.addEventListener('DOMContentLoaded', () => {
  // 2. Initialize Core ML Pipeline
  try {
    // Calculate global stats and scale dataset
    state.stats = calculateStats(state.rawDataset);
    state.standardizedDataset = standardizeDataset(state.rawDataset, state.stats);

    // Automatically run initial model evaluation (split validation)
    state.evaluationResults = evaluateModel(state.standardizedDataset, state.kValue, state.splitRatio);
    
    // Fit a deployment KNN model trained on the full dataset for playground inference
    state.predictorModel = new KNNClassifier(state.kValue);
    state.predictorModel.setTrainingData(state.standardizedDataset);

    console.log('--- BloomSphere ML Initialization ---');
    console.log('Dataset Mean/Std Dev computed.');
    console.log(`Automatic Train-Test split evaluation run (K = ${state.kValue}, split = ${state.splitRatio * 100}%):`);
    console.log(`Accuracy: ${(state.evaluationResults.accuracy * 100).toFixed(2)}%`);
    console.log(`F1 Score: ${(state.evaluationResults.f1Score * 100).toFixed(2)}%`);
    console.log('Confusion Matrix calculated:', state.evaluationResults.confusionMatrix);

    // Render metrics on the dashboard panel
    renderMetricsDashboard(state.evaluationResults);
  } catch (error) {
    console.error('Error during ML engine initialization:', error);
  }

  // 3. Initialize UI Components
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }

  // Set up input range sliders updates
  initSliders();

  // Scroll reveal intersections
  const revealElements = document.querySelectorAll('.scroll-reveal');
  const revealCallback = (entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        observer.unobserve(entry.target);
      }
    });
  };
  const revealObserver = new IntersectionObserver(revealCallback, {
    root: null,
    threshold: 0.15,
    rootMargin: '0px 0px -50px 0px'
  });
  revealElements.forEach(el => revealObserver.observe(el));

  // 4. Set Up SPA View Routing Toggles
  const startPredictBtns = [
    document.getElementById('btn-header-start'),
    document.getElementById('btn-hero-start')
  ];

  startPredictBtns.forEach(btn => {
    if (btn) {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        
        // Toggle view based on header state
        const isBackToHome = document.body.classList.contains('prediction-active') && e.currentTarget.id === 'btn-header-start';
        
        if (isBackToHome) {
          togglePredictionView(false);
        } else {
          togglePredictionView(true);
        }
      });
    }
  });

  // Logo click triggers home view restoration
  const logoBtn = document.querySelector('.logo-group');
  if (logoBtn) {
    logoBtn.addEventListener('click', (e) => {
      e.preventDefault();
      togglePredictionView(false);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // Nav link clicks restore home view before navigating to target hash
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      togglePredictionView(false);
    });
  });

  // 5. Prediction Execution Flow
  const executePredictionFlow = (e) => {
    e.preventDefault();
    console.log('executePredictionFlow started');

    try {
      // Validate all required DOM elements
      const requiredIds = [
        'prediction-result-card',
        'sepal-length',
        'sepal-width',
        'petal-length',
        'petal-width',
        'res-placeholder',
        'res-output',
        'res-error',
        'res-badge',
        'res-species-name',
        'res-scientific-name',
        'res-description',
        'res-icon-wrapper',
        'res-neighbors-list'
      ];
      
      const missingIds = [];
      requiredIds.forEach(id => {
        if (!document.getElementById(id)) {
          missingIds.push(id);
        }
      });

      if (missingIds.length > 0) {
        const errorMsg = `Required DOM elements missing: ${missingIds.map(id => `#${id}`).join(', ')}`;
        console.error(errorMsg);
        
        // Show error message in UI if card exists
        const card = document.getElementById('prediction-result-card');
        if (card) {
          const placeholder = document.getElementById('res-placeholder');
          const output = document.getElementById('res-output');
          if (placeholder) placeholder.classList.add('hidden');
          if (output) output.classList.add('hidden');
          
          let errorDiv = document.getElementById('res-error');
          if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'res-error';
            card.appendChild(errorDiv);
          }
          errorDiv.classList.remove('hidden');
          errorDiv.style.color = '#EF4444';
          errorDiv.style.fontFamily = 'Outfit, sans-serif';
          errorDiv.style.padding = '20px';
          errorDiv.style.textAlign = 'center';
          errorDiv.innerHTML = `
            <i data-lucide="alert-triangle" style="width: 40px; height: 40px; margin-bottom: 12px; display: inline-block;"></i>
            <h4 style="font-size: 18px; font-weight: 600; margin: 0 0 8px 0;">DOM Element Missing</h4>
            <p style="font-size: 13px; margin: 0; color: var(--text-muted);">The following required elements are missing: ${missingIds.map(id => `#${id}`).join(', ')}</p>
          `;
          card.className = 'glass-card result-card scroll-reveal revealed';
          card.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          card.style.boxShadow = '0 12px 40px rgba(239, 68, 68, 0.2)';
          if (typeof lucide !== 'undefined') {
            lucide.createIcons();
          }
        }
        throw new Error(errorMsg);
      }

      const card = document.getElementById('prediction-result-card');
      const sepalLengthEl = document.getElementById('sepal-length');
      const sepalWidthEl = document.getElementById('sepal-width');
      const petalLengthEl = document.getElementById('petal-length');
      const petalWidthEl = document.getElementById('petal-width');

      const rawQuery = {
        sepalLength: Number(sepalLengthEl.value),
        sepalWidth: Number(sepalWidthEl.value),
        petalLength: Number(petalLengthEl.value),
        petalWidth: Number(petalWidthEl.value)
      };

      console.log('Input values:', rawQuery);

      if (state.predictorModel === null) {
        throw new Error('KNN Predictor Model is not initialized.');
      }

      // Standardize query using dataset stats (Critical: Scale inputs!)
      const standardizedQuery = standardizeSample(rawQuery, state.stats);
      console.log('Standardized values:', standardizedQuery);

      // Run prediction
      const prediction = state.predictorModel.predict(standardizedQuery);
      console.log('Prediction returned:', prediction);

      // Verify KNNClassifier.predict() returns a valid species
      const validSpecies = ['setosa', 'versicolor', 'virginica'];
      if (!prediction || !prediction.species || !validSpecies.includes(prediction.species.toLowerCase())) {
        throw new Error(`Prediction did not return a valid species. Returned: ${prediction ? prediction.species : 'undefined'}`);
      }

      // If hidden, automatically reveal the result container
      if (card.classList.contains('hidden')) {
        card.classList.remove('hidden');
      }
      card.style.display = '';
      if (!card.classList.contains('revealed')) {
        card.classList.add('revealed');
      }
      console.log('Result card revealed');

      // Render predicted result card in UI
      console.log('renderPredictionResult called');
      renderPredictionResult(prediction, rawQuery);

      // Smoothly scroll the card into center view
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (error) {
      console.error('Prediction failed with error:', error);

      // If prediction fails, show "Prediction Error - Check Console"
      const placeholder = document.getElementById('res-placeholder');
      const output = document.getElementById('res-output');
      const card = document.getElementById('prediction-result-card');

      if (card) {
        if (placeholder) placeholder.classList.add('hidden');
        if (output) output.classList.add('hidden');

        // Reveal card if hidden
        if (card.classList.contains('hidden')) {
          card.classList.remove('hidden');
        }
        card.style.display = '';
        if (!card.classList.contains('revealed')) {
          card.classList.add('revealed');
        }

        // Show error message
        let errorDiv = document.getElementById('res-error');
        if (!errorDiv) {
          errorDiv = document.createElement('div');
          errorDiv.id = 'res-error';
          errorDiv.style.color = '#EF4444';
          errorDiv.style.fontFamily = 'Outfit, sans-serif';
          errorDiv.style.padding = '20px';
          errorDiv.style.textAlign = 'center';
          card.appendChild(errorDiv);
        }
        errorDiv.classList.remove('hidden');
        errorDiv.innerHTML = `
          <i data-lucide="alert-triangle" style="width: 40px; height: 40px; margin-bottom: 12px; display: inline-block;"></i>
          <h4 style="font-size: 18px; font-weight: 600; margin: 0 0 8px 0;">Prediction Error - Check Console</h4>
          <p style="font-size: 13px; margin: 0; color: var(--text-muted);">${error.message}</p>
        `;
        card.className = 'glass-card result-card scroll-reveal revealed';
        card.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        card.style.boxShadow = '0 12px 40px rgba(239, 68, 68, 0.2)';

        if (typeof lucide !== 'undefined') {
          lucide.createIcons();
        }
      }
    }
  };

  // Attach only form submission event listener to prevent duplicate triggers
  const predictionForm = document.getElementById('prediction-form');
  if (predictionForm) {
    predictionForm.addEventListener('submit', executePredictionFlow);
  }
});
