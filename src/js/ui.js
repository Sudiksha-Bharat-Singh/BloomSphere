/**
 * BloomSphere UI Module
 * Handles DOM manipulation, range sliders, dataset table rendering, circular gauges,
 * confusion matrix coloring, and custom prediction confidence breakdowns.
 */

// Species descriptions matching classification classes
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
 * Initializes sliders and ties inputs to value displays.
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
 * Helper to update SVG progress circle dash offset.
 */
function setCircleProgress(id, percentage) {
  const circle = document.getElementById(id);
  if (circle) {
    const circumference = 251.2; // 2 * pi * r (40)
    const offset = circumference - (percentage / 100) * circumference;
    circle.style.strokeDashoffset = offset;
  }
}

/**
 * Updates the Model Evaluation dashboard circular metrics.
 * @param {Object} results - Evaluation results metrics.
 */
export function renderMetricsDashboard(results) {
  const accuracyEl = document.getElementById('metric-accuracy');
  const precisionEl = document.getElementById('metric-precision');
  const recallEl = document.getElementById('metric-recall');
  const f1El = document.getElementById('metric-f1-score');

  const accPct = results.accuracy * 100;
  const precPct = results.precision * 100;
  const recPct = results.recall * 100;
  const f1Pct = results.f1Score * 100;

  if (accuracyEl) accuracyEl.textContent = `${accPct.toFixed(1)}%`;
  if (precisionEl) precisionEl.textContent = `${precPct.toFixed(1)}%`;
  if (recallEl) recallEl.textContent = `${recPct.toFixed(1)}%`;
  if (f1El) f1El.textContent = `${f1Pct.toFixed(1)}%`;

  // Animate circles
  setCircleProgress('circle-accuracy', accPct);
  setCircleProgress('circle-precision', precPct);
  setCircleProgress('circle-recall', recPct);
  setCircleProgress('circle-f1', f1Pct);
}

/**
 * Renders the 3x3 Confusion Matrix with actual evaluation values.
 * @param {Array} matrix - 3x3 matrix from model evaluation.
 */
export function renderConfusionMatrix(matrix) {
  if (!matrix || matrix.length !== 3) return;

  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const cell = document.getElementById(`cm-${r}-${c}`);
      if (cell) {
        const val = matrix[r][c];
        cell.textContent = val;

        // Reset classes
        cell.className = 'matrix-value-cell';
        if (r === c) {
          // Diagonal: correct predictions
          cell.classList.add('matrix-correct');
        } else if (val > 0) {
          // Off-diagonal: incorrect predictions
          cell.classList.add('matrix-incorrect');
        }
      }
    }
  }
}

/**
 * Renders the prediction outcome on the Result Card inside Prediction Workspace.
 * @param {Object} prediction - Output of KNN classifier predict().
 * @param {Object} rawQuery - Unscaled query parameters entered by user.
 */
export function renderPredictionResult(prediction, rawQuery) {
  console.log('Rendering prediction result inside UI module...');
  
  const placeholder = document.getElementById('res-placeholder');
  const output = document.getElementById('res-output');
  const card = document.getElementById('prediction-result-card');
  const errorDiv = document.getElementById('res-error');

  if (!card) {
    console.error('Prediction result container (#prediction-result-card) is missing in DOM.');
    return;
  }
  if (!placeholder || !output) {
    console.error('Required DOM components inside result container missing:', { placeholder, output });
    return;
  }

  // Reveal card
  card.classList.remove('hidden');
  card.style.display = 'block';

  // Hide any previous errors
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

  // Restyle border and glows based on species
  card.className = 'glass-card result-card scroll-reveal revealed';
  card.classList.add(info.glowClass);
  card.style.borderColor = '';
  card.style.boxShadow = '';

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

  // Draw Confidence Breakdown
  const confidenceList = document.getElementById('res-confidence-list');
  if (confidenceList && prediction.votes) {
    confidenceList.innerHTML = '';
    const k = prediction.neighbors.length || 5;
    const classes = ['setosa', 'versicolor', 'virginica'];

    classes.forEach(sp => {
      const count = prediction.votes[sp] || 0;
      const percentage = (count / k) * 100;

      const item = document.createElement('div');
      item.className = 'confidence-item';
      item.innerHTML = `
        <div class="confidence-info">
          <span class="confidence-name">${sp}</span>
          <span class="confidence-pct">${percentage.toFixed(0)}%</span>
        </div>
        <div class="confidence-bar-bg">
          <div class="confidence-bar-fill" style="width: ${percentage}%"></div>
        </div>
      `;
      confidenceList.appendChild(item);
    });
  }

  // Render neighbors breakdown
  const neighborsList = document.getElementById('res-neighbors-list');
  if (neighborsList) {
    neighborsList.innerHTML = '';
    
    prediction.neighbors.forEach((neighbor, idx) => {
      const neighborSpecies = neighbor.species.charAt(0).toUpperCase() + neighbor.species.slice(1);
      const neighborItem = document.createElement('div');
      neighborItem.className = 'neighbor-item';
      
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
 * Initializes the Dataset Explorer and Full Dataset Modal.
 * @param {Array} dataset - Original raw dataset array.
 */
export function initDatasetExplorer(dataset) {
  const tableBody = document.getElementById('dataset-table-body');
  const modalTableBody = document.getElementById('dataset-modal-table-body');
  const searchInput = document.getElementById('dataset-search');
  const countBadge = document.getElementById('dataset-count');
  
  const viewAllBtn = document.getElementById('btn-view-all-dataset');
  const closeModalBtn = document.getElementById('btn-close-modal');
  const modalOverlay = document.getElementById('dataset-modal-overlay');

  if (!tableBody || !dataset) return;

  // Species classes mapping
  const getSpeciesClass = (sp) => {
    const s = sp.toLowerCase();
    if (s === 'setosa') return 'species-setosa-cell';
    if (s === 'versicolor') return 'species-versicolor-cell';
    if (s === 'virginica') return 'species-virginica-cell';
    return '';
  };

  // Render function
  const renderTables = (data) => {
    tableBody.innerHTML = '';
    const first10 = data.slice(0, 10);
    first10.forEach(sample => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${sample.sepalLength.toFixed(1)}</td>
        <td>${sample.sepalWidth.toFixed(1)}</td>
        <td>${sample.petalLength.toFixed(1)}</td>
        <td>${sample.petalWidth.toFixed(1)}</td>
        <td class="dataset-species-cell ${getSpeciesClass(sample.species)}">${sample.species}</td>
      `;
      tableBody.appendChild(tr);
    });

    if (modalTableBody) {
      modalTableBody.innerHTML = '';
      data.forEach(sample => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${sample.sepalLength.toFixed(1)}</td>
          <td>${sample.sepalWidth.toFixed(1)}</td>
          <td>${sample.petalLength.toFixed(1)}</td>
          <td>${sample.petalWidth.toFixed(1)}</td>
          <td class="dataset-species-cell ${getSpeciesClass(sample.species)}">${sample.species}</td>
        `;
        modalTableBody.appendChild(tr);
      });
    }

    if (countBadge) {
      countBadge.textContent = `Showing ${data.length} of ${dataset.length} records`;
    }
  };

  // Initial draw
  renderTables(dataset);

  // Search input handler
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase().trim();
      const filtered = dataset.filter(sample => {
        const speciesMatch = sample.species.toLowerCase().includes(query);
        const sepalMatch = sample.sepalLength.toString().includes(query);
        const petalMatch = sample.petalLength.toString().includes(query);
        return speciesMatch || sepalMatch || petalMatch;
      });
      renderTables(filtered);
    });
  }

  // Modal event hooks
  if (viewAllBtn && modalOverlay) {
    viewAllBtn.addEventListener('click', (e) => {
      e.preventDefault();
      modalOverlay.classList.add('active');
    });
  }
  if (closeModalBtn && modalOverlay) {
    closeModalBtn.addEventListener('click', (e) => {
      e.preventDefault();
      modalOverlay.classList.remove('active');
    });
  }
  if (modalOverlay) {
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) {
        modalOverlay.classList.remove('active');
      }
    });
  }
}

/**
 * Toggles visibility or scrolls to components.
 * @param {boolean} active - Scroll trigger.
 */
export function togglePredictionView(active) {
  const predictCard = document.getElementById('prediction-workspace-card');
  if (predictCard) {
    predictCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    predictCard.style.outline = '2px solid var(--color-highlight)';
    setTimeout(() => {
      predictCard.style.outline = 'none';
    }, 1500);
  }
}
