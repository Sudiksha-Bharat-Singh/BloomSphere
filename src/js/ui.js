/**
 * BloomSphere UI Module
 * Handles DOM manipulation, view toggles, range sliders, results and metrics rendering.
 */

// Species descriptions and classes matching Phase 3
const speciesDetails = {
  setosa: {
    name: 'Iris Setosa',
    scientific: 'Iris setosa',
    badge: 'Class 1',
    description: 'Known for its small petals and easy separability from other species.',
    glowClass: 'prediction-setosa',
    icon: 'flower-2'
  },
  versicolor: {
    name: 'Iris Versicolor',
    scientific: 'Iris versicolor',
    badge: 'Class 2',
    description: 'A medium-sized species with balanced petal and sepal dimensions.',
    glowClass: 'prediction-versicolor',
    icon: 'flower'
  },
  virginica: {
    name: 'Iris Virginica',
    scientific: 'Iris virginica',
    badge: 'Class 3',
    description: 'The largest Iris species with longer petals and broader measurements.',
    glowClass: 'prediction-virginica',
    icon: 'flower'
  }
};

/**
 * Initializes sliders and ties inputs to value counters.
 */
export function initSliders() {
  const sliders = [
    { id: 'sepal-length', valId: 'val-sepal-length' },
    { id: 'sepal-width', valId: 'val-sepal-width' },
    { id: 'petal-length', valId: 'val-petal-length' },
    { id: 'petal-width', valId: 'val-petal-width' }
  ];

  sliders.forEach(slider => {
    const inputEl = document.getElementById(slider.id);
    const displayEl = document.getElementById(slider.valId);
    if (inputEl && displayEl) {
      inputEl.addEventListener('input', (e) => {
        displayEl.textContent = Number(e.target.value).toFixed(1);
      });
    }
  });
}

/**
 * Renders the prediction outcome on the Result Card.
 * @param {Object} prediction - Output of KNN classifier predict().
 * @param {Object} rawQuery - Unscaled query parameters entered by user.
 */
export function renderPredictionResult(prediction, rawQuery) {
  console.log('Rendering prediction result inside UI module...');
  
  const placeholder = document.getElementById('res-placeholder');
  const output = document.getElementById('res-output');
  const card = document.getElementById('prediction-result-card');
  const errorDiv = document.getElementById('res-error');

  // Verify the prediction result container exists in the DOM
  if (!card) {
    console.error('Prediction result container (#prediction-result-card) does not exist in the DOM.');
    return;
  }
  if (!placeholder || !output) {
    console.error('DOM elements inside result container missing:', { placeholder, output });
    return;
  }

  // If hidden, automatically reveal it after prediction
  if (card.classList.contains('hidden')) {
    card.classList.remove('hidden');
  }
  card.style.display = '';
  if (!card.classList.contains('revealed')) {
    card.classList.add('revealed');
  }

  // If error was previously visible, hide it
  if (errorDiv) {
    errorDiv.classList.add('hidden');
  }

  const speciesName = prediction.species.toLowerCase();
  const info = speciesDetails[speciesName] || {
    name: 'Unknown Species',
    scientific: 'N/A',
    badge: 'Class -',
    description: 'The measurements did not resolve to a standard species class.',
    glowClass: 'prediction-unknown',
    icon: 'help-circle'
  };

  // Remove previous result glows and empty state, preserving structural classes
  card.classList.remove('empty-state', 'prediction-setosa', 'prediction-versicolor', 'prediction-virginica', 'prediction-unknown');
  card.style.borderColor = '';
  card.style.boxShadow = '';
  card.classList.add(info.glowClass);

  // Set values
  document.getElementById('res-badge').textContent = info.badge;
  document.getElementById('res-species-name').textContent = info.name;
  document.getElementById('res-scientific-name').textContent = info.scientific;
  document.getElementById('res-description').textContent = info.description;

  // Set icon
  const iconWrapper = document.getElementById('res-icon-wrapper');
  if (iconWrapper) {
    iconWrapper.innerHTML = `<i data-lucide="${info.icon}"></i>`;
  }

  // Render neighbors breakdown
  const neighborsList = document.getElementById('res-neighbors-list');
  if (neighborsList) {
    neighborsList.innerHTML = '';
    
    // Take the nearest neighbors and display their distance and class
    prediction.neighbors.forEach((neighbor, idx) => {
      const neighborSpecies = neighbor.species.charAt(0).toUpperCase() + neighbor.species.slice(1);
      const neighborItem = document.createElement('div');
      neighborItem.className = 'neighbor-item';
      
      // Highlight if same species
      const isMatch = neighbor.species.toLowerCase() === speciesName;
      const matchBadge = isMatch ? '<span class="neighbor-match">Vote</span>' : '';

      neighborItem.innerHTML = `
        <span class="neighbor-index">#${idx + 1}</span>
        <span class="neighbor-name">${neighborSpecies}</span>
        <span class="neighbor-distance">d = ${neighbor.distance.toFixed(3)}</span>
        ${matchBadge}
      `;
      neighborsList.appendChild(neighborItem);
    });
  }

  // Toggle visible elements
  placeholder.classList.add('hidden');
  output.classList.remove('hidden');

  // Ensure card is revealed
  card.classList.add('revealed');

  // Trigger Lucide to parse new icons
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }

  // Diagnostic Logs
  console.log('--- DIAGNOSTIC DOM AUDIT ---');
  const ids = ['prediction-result-card', 'res-placeholder', 'res-output', 'res-icon-wrapper', 'res-species-name', 'res-scientific-name', 'res-badge', 'res-description', 'res-neighbors-list'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el) {
      console.log(`Element #${id} is NULL!`);
    } else {
      const styles = window.getComputedStyle(el);
      console.log(`Element #${id}:`, {
        display: styles.display,
        visibility: styles.visibility,
        opacity: styles.opacity,
        color: styles.color,
        textContent: el.textContent.trim(),
        innerHTML: el.innerHTML.trim()
      });
    }
  });
  console.log('----------------------------');
}

/**
 * Updates the Model Evaluation dashboard.
 * @param {Object} results - Evaluation results metrics.
 */
export function renderMetricsDashboard(results) {
  const accuracyEl = document.getElementById('metric-accuracy');
  const precisionEl = document.getElementById('metric-precision');
  const recallEl = document.getElementById('metric-recall');
  const f1El = document.getElementById('metric-f1-score');

  if (accuracyEl) accuracyEl.textContent = `${(results.accuracy * 100).toFixed(1)}%`;
  if (precisionEl) precisionEl.textContent = `${(results.precision * 100).toFixed(1)}%`;
  if (recallEl) recallEl.textContent = `${(results.recall * 100).toFixed(1)}%`;
  if (f1El) f1El.textContent = `${(results.f1Score * 100).toFixed(1)}%`;
}

/**
 * Toggles visibility between Home layout and Prediction Page.
 * @param {boolean} active - Set to true to display Predict page, false for Home.
 */
export function togglePredictionView(active) {
  const body = document.body;
  const predictSection = document.getElementById('prediction');
  const homeSections = [
    document.querySelector('.section-hero'),
    document.querySelector('.section-stats'),
    document.querySelector('.section-about'),
    document.querySelector('.section-species'),
    document.querySelector('.section-workflow')
  ];

  const headerBtn = document.getElementById('btn-header-start');

  if (active) {
    body.classList.add('prediction-active');
    if (predictSection) predictSection.classList.remove('hidden');
    homeSections.forEach(section => section && section.classList.add('hidden'));
    
    // Force reveal elements inside prediction workspace to bypass intersection delays
    if (predictSection) {
      const revealElems = predictSection.querySelectorAll('.scroll-reveal');
      revealElems.forEach(el => el.classList.add('revealed'));
    }

    // Scroll to workspace top
    window.scrollTo({ top: 0, behavior: 'instant' });

    // Update sticky button to Back to Home
    if (headerBtn) {
      headerBtn.innerHTML = `
        <i data-lucide="arrow-left"></i>
        <span>Back to Home</span>
      `;
    }
  } else {
    body.classList.remove('prediction-active');
    if (predictSection) predictSection.classList.add('hidden');
    homeSections.forEach(section => section && section.classList.remove('hidden'));

    if (headerBtn) {
      headerBtn.innerHTML = `
        <span>Start Prediction</span>
        <i data-lucide="arrow-right"></i>
      `;
    }
  }

  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }
}
