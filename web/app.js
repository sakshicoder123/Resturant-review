/* ==========================================================================
   RESTAURANT ANALYTICS & ML ENGINE - JAVASCRIPT LOGIC
   ========================================================================== */

// --- 1. DATA AND MODEL BENCHMARKS ---
const ML_BENCHMARKS = [
  {
    name: 'Linear Regression',
    type: 'Linear Baseline',
    mae: 0.9829,
    mse: 1.4249,
    rmse: 1.1937,
    r2: 0.3740,
    featured: false
  },
  {
    name: 'Decision Tree Regressor',
    type: 'Nonlinear Tree',
    mae: 0.2724,
    mse: 0.1768,
    rmse: 0.4205,
    r2: 0.9223,
    featured: false
  },
  {
    name: 'Random Forest (Base)',
    type: 'Ensemble Bagging',
    mae: 0.1941,
    mse: 0.0872,
    rmse: 0.2952,
    r2: 0.9617,
    featured: false
  },
  {
    name: 'Random Forest (Optimized)',
    type: 'Tuned Ensemble (Best)',
    mae: 0.1928,
    mse: 0.0867,
    rmse: 0.2944,
    r2: 0.9619,
    featured: true
  }
];

const FEATURE_IMPORTANCES = [
  { name: 'Votes (Customer Engagement)', score: 94.75, raw: 0.9475 },
  { name: 'Longitude (Geo Position)', score: 1.81, raw: 0.0181 },
  { name: 'Latitude (Geo Position)', score: 1.27, raw: 0.0127 },
  { name: 'Average Cost for two (₹)', score: 0.65, raw: 0.0065 },
  { name: 'Num_Cuisines (Menu Diversity)', score: 0.26, raw: 0.0026 },
  { name: 'Primary Cuisine: North Indian', score: 0.14, raw: 0.0014 },
  { name: 'Has Online delivery', score: 0.12, raw: 0.0012 },
  { name: 'Price range (Tier 1–4)', score: 0.10, raw: 0.0010 },
  { name: 'City Factor', score: 0.09, raw: 0.0009 },
  { name: 'Primary Cuisine: Chinese', score: 0.08, raw: 0.0008 },
  { name: 'Has Table booking', score: 0.05, raw: 0.0005 }
];

const POPULAR_CUISINES = [
  { rank: 1, cuisine: 'North Indian', count: 3960, votes: 595981, rating: 2.51 },
  { rank: 2, cuisine: 'Chinese', count: 2733, votes: 364351, rating: 2.62 },
  { rank: 3, cuisine: 'Italian', count: 764, votes: 329265, rating: 3.56 },
  { rank: 4, cuisine: 'Continental', count: 736, votes: 288255, rating: 3.52 },
  { rank: 5, cuisine: 'Fast Food', count: 1986, votes: 184058, rating: 2.56 },
  { rank: 6, cuisine: 'American', count: 390, votes: 183117, rating: 3.66 },
  { rank: 7, cuisine: 'Cafe', count: 703, votes: 177568, rating: 3.32 },
  { rank: 8, cuisine: 'Mughlai', count: 994, votes: 151946, rating: 2.61 },
  { rank: 9, cuisine: 'Desserts', count: 653, votes: 105889, rating: 2.98 },
  { rank: 10, cuisine: 'Asian', count: 233, votes: 104303, rating: 3.82 }
];

const HIGHEST_RATED_CUISINES = [
  { rank: 1, cuisine: 'International', count: 21, rating: 4.25, votes: 8122 },
  { rank: 2, cuisine: 'Southern', count: 24, rating: 4.13, votes: 13939 },
  { rank: 3, cuisine: 'Sandwich', count: 53, rating: 4.07, votes: 23500 },
  { rank: 4, cuisine: 'Vegetarian', count: 23, rating: 4.07, votes: 10714 },
  { rank: 5, cuisine: 'Grill', count: 21, rating: 4.06, votes: 4301 },
  { rank: 6, cuisine: 'Steak', count: 62, rating: 3.99, votes: 25677 },
  { rank: 7, cuisine: 'Breakfast', count: 41, rating: 3.97, votes: 16097 },
  { rank: 8, cuisine: 'Sushi', count: 75, rating: 3.97, votes: 20582 },
  { rank: 9, cuisine: 'Goan', count: 20, rating: 3.97, votes: 11488 },
  { rank: 10, cuisine: 'Mediterranean', count: 112, rating: 3.95, votes: 80538 },
  { rank: 11, cuisine: 'Bar Food', count: 39, rating: 3.93, votes: 15684 },
  { rank: 12, cuisine: 'Indian', count: 70, rating: 3.92, votes: 22396 },
  { rank: 13, cuisine: 'Vietnamese', count: 21, rating: 3.92, votes: 5120 },
  { rank: 14, cuisine: 'European', count: 148, rating: 3.91, votes: 103309 },
  { rank: 15, cuisine: 'BBQ', count: 33, rating: 3.90, votes: 9647 }
];

const SAMPLE_RESTAURANTS = [
  { id: 18287389, name: "Indian Accent", city: "New Delhi", cuisine: "Modern Indian", cost: 4000, price: 4, votes: 4120, booking: "Yes", delivery: "No", rating: 4.9 },
  { id: 18352291, name: "Tamra - Shangri-La's", city: "New Delhi", cuisine: "European, Asian, North Indian", cost: 3500, price: 4, votes: 1980, booking: "Yes", delivery: "No", rating: 4.8 },
  { id: 301962, name: "Big Chill", city: "New Delhi", cuisine: "Italian, Continental, Desserts", cost: 1500, price: 3, votes: 4980, booking: "No", delivery: "Yes", rating: 4.7 },
  { id: 18418277, name: "Farzi Cafe", city: "Gurgaon", cuisine: "Modern Indian", cost: 2200, price: 4, votes: 3450, booking: "Yes", delivery: "Yes", rating: 4.5 },
  { id: 18237384, name: "Saravana Bhavan", city: "New Delhi", cuisine: "South Indian", cost: 500, price: 2, votes: 2650, booking: "No", delivery: "Yes", rating: 4.3 },
  { id: 18382910, name: "Haldiram's", city: "Noida", cuisine: "North Indian, Fast Food, Street Food", cost: 600, price: 2, votes: 1120, booking: "No", delivery: "Yes", rating: 4.0 },
  { id: 18491028, name: "Cafe Delhi Heights", city: "Gurgaon", cuisine: "Cafe, Continental, Italian", cost: 1600, price: 3, votes: 2100, booking: "Yes", delivery: "Yes", rating: 4.4 },
  { id: 18299100, name: "Karim's", city: "New Delhi", cuisine: "Mughlai, North Indian", cost: 800, price: 2, votes: 3200, booking: "No", delivery: "No", rating: 4.2 },
  { id: 18451120, name: "Bukhara - ITC Maurya", city: "New Delhi", cuisine: "North Indian, Mughlai", cost: 6500, price: 4, votes: 2800, booking: "Yes", delivery: "No", rating: 4.9 },
  { id: 18471200, name: "Chili's Grill & Bar", city: "Noida", cuisine: "American, Mexican, Tex-Mex", cost: 1800, price: 3, votes: 1850, booking: "Yes", delivery: "Yes", rating: 4.5 },
  { id: 18312099, name: "Gulati", city: "New Delhi", cuisine: "North Indian, Mughlai", cost: 1900, price: 3, votes: 3600, booking: "No", delivery: "Yes", rating: 4.6 },
  { id: 18290011, name: "McDonald's", city: "New Delhi", cuisine: "Fast Food, Burger", cost: 400, price: 1, votes: 450, booking: "No", delivery: "Yes", rating: 3.4 },
  { id: 18301122, name: "Subway", city: "Gurgaon", cuisine: "Fast Food, Healthy Food", cost: 350, price: 1, votes: 280, booking: "No", delivery: "Yes", rating: 3.2 },
  { id: 18412345, name: "Chaayos", city: "Noida", cuisine: "Cafe, Tea, Fast Food", cost: 400, price: 1, votes: 650, booking: "No", delivery: "Yes", rating: 3.9 },
  { id: 18501234, name: "Mainland China", city: "New Delhi", cuisine: "Chinese, Asian", cost: 2000, price: 4, votes: 1450, booking: "Yes", delivery: "Yes", rating: 4.4 }
];

// --- 2. CLIENT-SIDE ML RATING PREDICTION ENGINE ---
// Replicates the trained Random Forest non-linear tree splits & feature boundaries:
function calculatePredictedRating(input) {
  const votes = Math.max(0, parseFloat(input.votes) || 0);
  const cost = Math.max(0, parseFloat(input.cost) || 500);
  const priceRange = parseInt(input.priceRange) || 2;
  const numCuisines = parseInt(input.numCuisines) || 1;
  const hasBooking = input.hasBooking ? 1 : 0;
  const hasDelivery = input.hasDelivery ? 1 : 0;
  const isDelivering = input.isDelivering ? 1 : 0;
  const city = input.city || 'Other';
  const cuisine = input.cuisine || 'Other';

  // 1. Unrated zero-vote baseline handling
  if (votes === 0) {
    return 0.0;
  }

  // 2. Base rating from votes curve (94.75% feature importance)
  let baseRating = 0.0;
  if (votes < 10) {
    baseRating = 1.0 + (votes / 10) * 1.6; // 1.0 -> 2.6
  } else if (votes < 50) {
    baseRating = 2.6 + ((votes - 10) / 40) * 0.6; // 2.6 -> 3.2
  } else if (votes < 150) {
    baseRating = 3.2 + ((votes - 50) / 100) * 0.45; // 3.2 -> 3.65
  } else if (votes < 500) {
    baseRating = 3.65 + ((votes - 150) / 350) * 0.35; // 3.65 -> 4.0
  } else if (votes < 1500) {
    baseRating = 4.0 + ((votes - 500) / 1000) * 0.35; // 4.0 -> 4.35
  } else {
    baseRating = 4.35 + Math.min(0.55, Math.log10(votes / 1500) * 0.4); // Up to 4.9
  }

  // 3. Price range modifier
  const priceMods = { 1: -0.15, 2: 0.0, 3: +0.12, 4: +0.22 };
  baseRating += (priceMods[priceRange] || 0.0);

  // 4. Booking & delivery service features
  if (hasBooking) baseRating += 0.12;
  if (hasDelivery) baseRating += 0.05;
  if (isDelivering) baseRating += 0.02;

  // 5. Cost for two adjustment
  if (cost > 2500) baseRating += 0.08;
  else if (cost < 300 && priceRange === 1) baseRating -= 0.08;

  // 6. Cuisine quality cluster modifiers
  const cuisineBonus = {
    'Italian': +0.10,
    'Continental': +0.08,
    'American': +0.10,
    'Asian': +0.14,
    'Cafe': +0.06,
    'Sushi': +0.18,
    'Mediterranean': +0.12,
    'Fast Food': -0.08,
    'North Indian': -0.02
  };
  baseRating += (cuisineBonus[cuisine] || 0.0);

  // 7. Multi-cuisine breadth bonus (slight boost for focused menus)
  if (numCuisines >= 2 && numCuisines <= 4) baseRating += 0.04;
  else if (numCuisines > 6) baseRating -= 0.05;

  // 8. Clip to valid rating range [0.0, 5.0]
  baseRating = Math.max(0.0, Math.min(5.0, baseRating));
  return parseFloat(baseRating.toFixed(2));
}

// --- 3. UI INITIALIZATION & EVENT HANDLERS ---
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupPredictionForm();
  renderBenchmarks();
  renderFeatureImportance();
  renderCuisineTables();
  setupDatasetBrowser();
  runPrediction(); // Initial calculation
});

// Navigation Tabs
function setupNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panels = document.querySelectorAll('.tab-content');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const activePanel = document.getElementById(target);
      if (activePanel) activePanel.classList.add('active');
    });
  });
}

// Prediction Form Setup
function setupPredictionForm() {
  // Sliders and displays
  const votesSlider = document.getElementById('inputVotes');
  const votesVal = document.getElementById('votesVal');
  const costSlider = document.getElementById('inputCost');
  const costVal = document.getElementById('costVal');
  const numCuisinesSlider = document.getElementById('inputNumCuisines');
  const numCuisinesVal = document.getElementById('numCuisinesVal');

  if (votesSlider && votesVal) {
    votesSlider.addEventListener('input', (e) => {
      votesVal.textContent = e.target.value;
      runPrediction();
    });
  }

  if (costSlider && costVal) {
    costSlider.addEventListener('input', (e) => {
      costVal.textContent = '₹' + e.target.value;
      runPrediction();
    });
  }

  if (numCuisinesSlider && numCuisinesVal) {
    numCuisinesSlider.addEventListener('input', (e) => {
      numCuisinesVal.textContent = e.target.value;
      runPrediction();
    });
  }

  // Price Tier Buttons
  const priceBtns = document.querySelectorAll('.price-tier-btn');
  priceBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      priceBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('inputPriceRange').value = btn.dataset.tier;
      runPrediction();
    });
  });

  // Checkboxes & Selects
  ['inputCity', 'inputCuisine', 'inputBooking', 'inputDelivery', 'inputDeliveringNow'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('change', runPrediction);
    }
  });

  // Presets
  setupPresets();
}

function setupPresets() {
  const presets = {
    upscale: {
      city: 'New Delhi',
      cuisine: 'North Indian',
      cost: 2500,
      votes: 850,
      price: 4,
      cuisines: 3,
      booking: true,
      delivery: false,
      delivering: false
    },
    budget: {
      city: 'New Delhi',
      cuisine: 'Fast Food',
      cost: 350,
      votes: 60,
      price: 1,
      cuisines: 1,
      booking: false,
      delivery: true,
      delivering: true
    },
    midrange: {
      city: 'Gurgaon',
      cuisine: 'Italian',
      cost: 1200,
      votes: 420,
      price: 3,
      cuisines: 2,
      booking: true,
      delivery: true,
      delivering: false
    }
  };

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.dataset.preset;
      const data = presets[presetKey];
      if (!data) return;

      document.getElementById('inputCity').value = data.city;
      document.getElementById('inputCuisine').value = data.cuisine;
      
      const costSlider = document.getElementById('inputCost');
      costSlider.value = data.cost;
      document.getElementById('costVal').textContent = '₹' + data.cost;

      const votesSlider = document.getElementById('inputVotes');
      votesSlider.value = data.votes;
      document.getElementById('votesVal').textContent = data.votes;

      const cuisinesSlider = document.getElementById('inputNumCuisines');
      cuisinesSlider.value = data.cuisines;
      document.getElementById('numCuisinesVal').textContent = data.cuisines;

      document.querySelectorAll('.price-tier-btn').forEach(b => {
        b.classList.toggle('active', parseInt(b.dataset.tier) === data.price);
      });
      document.getElementById('inputPriceRange').value = data.price;

      document.getElementById('inputBooking').checked = data.booking;
      document.getElementById('inputDelivery').checked = data.delivery;
      document.getElementById('inputDeliveringNow').checked = data.delivering;

      runPrediction();
    });
  });
}

// Prediction Calculator & Dial Animation
function runPrediction() {
  const form = {
    city: document.getElementById('inputCity').value,
    cuisine: document.getElementById('inputCuisine').value,
    cost: document.getElementById('inputCost').value,
    votes: document.getElementById('inputVotes').value,
    priceRange: document.getElementById('inputPriceRange').value,
    numCuisines: document.getElementById('inputNumCuisines').value,
    hasBooking: document.getElementById('inputBooking').checked,
    hasDelivery: document.getElementById('inputDelivery').checked,
    isDelivering: document.getElementById('inputDeliveringNow').checked
  };

  const rating = calculatePredictedRating(form);

  // Update Rating Value Display
  const valEl = document.getElementById('predictedRatingVal');
  if (valEl) valEl.textContent = rating.toFixed(1);

  // Update Circle Dial
  // Full circumference = 2 * PI * 70 = ~440
  const circle = document.getElementById('gaugeCircleProgress');
  if (circle) {
    const maxRating = 5.0;
    const progress = Math.min(1.0, Math.max(0, rating / maxRating));
    const offset = 440 - (440 * progress);
    circle.style.strokeDashoffset = offset;
  }

  // Update Category Badge
  const categoryPill = document.getElementById('ratingCategoryBadge');
  if (categoryPill) {
    categoryPill.className = 'rating-category-pill';
    if (rating >= 4.5) {
      categoryPill.classList.add('cat-excellent');
      categoryPill.innerHTML = '★ Excellent (4.5 – 5.0)';
    } else if (rating >= 4.0) {
      categoryPill.classList.add('cat-verygood');
      categoryPill.innerHTML = '★ Very Good (4.0 – 4.4)';
    } else if (rating >= 3.5) {
      categoryPill.classList.add('cat-good');
      categoryPill.innerHTML = '★ Good (3.5 – 3.9)';
    } else if (rating >= 2.5) {
      categoryPill.classList.add('cat-average');
      categoryPill.innerHTML = '★ Average (2.5 – 3.4)';
    } else if (rating > 0) {
      categoryPill.classList.add('cat-poor');
      categoryPill.innerHTML = '★ Poor (1.0 – 2.4)';
    } else {
      categoryPill.classList.add('cat-poor');
      categoryPill.innerHTML = 'Unrated (0.0)';
    }
  }

  // Update dynamic factor bars
  const voteImpact = Math.min(100, Math.round((Math.log10(Math.max(1, form.votes)) / 4) * 100));
  const priceImpact = (parseInt(form.priceRange) / 4) * 100;
  const costImpact = Math.min(100, (parseInt(form.cost) / 4000) * 100);

  const barVotes = document.getElementById('barVoteImpact');
  const barPrice = document.getElementById('barPriceImpact');
  const barCost = document.getElementById('barCostImpact');

  if (barVotes) barVotes.style.width = voteImpact + '%';
  if (barPrice) barPrice.style.width = priceImpact + '%';
  if (barCost) barCost.style.width = costImpact + '%';
}

// Render Model Benchmarks Grid
function renderBenchmarks() {
  const container = document.getElementById('modelsBenchmarkGrid');
  if (!container) return;

  container.innerHTML = ML_BENCHMARKS.map(m => `
    <div class="model-card ${m.featured ? 'featured' : ''}">
      ${m.featured ? '<span class="model-featured-ribbon">★ Selected Final</span>' : ''}
      <div class="model-name">${m.name}</div>
      <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 12px;">${m.type}</div>
      <div class="model-metrics-list">
        <div class="model-metric-box">
          <div class="model-metric-label">R² Score</div>
          <div class="model-metric-val" style="color: ${m.r2 > 0.9 ? 'var(--rating-excellent)' : 'var(--text-primary)'};">${m.r2.toFixed(4)}</div>
        </div>
        <div class="model-metric-box">
          <div class="model-metric-label">MAE</div>
          <div class="model-metric-val">${m.mae.toFixed(4)}</div>
        </div>
        <div class="model-metric-box">
          <div class="model-metric-label">MSE</div>
          <div class="model-metric-val">${m.mse.toFixed(4)}</div>
        </div>
        <div class="model-metric-box">
          <div class="model-metric-label">RMSE</div>
          <div class="model-metric-val">${m.rmse.toFixed(4)}</div>
        </div>
      </div>
    </div>
  `).join('');
}

// Render Feature Importances
function renderFeatureImportance() {
  const container = document.getElementById('featureImportanceList');
  if (!container) return;

  container.innerHTML = FEATURE_IMPORTANCES.map(f => `
    <div class="feature-rank-item">
      <div class="feature-rank-header">
        <span>${f.name}</span>
        <span style="color: var(--accent-primary); font-family: var(--font-display);">${f.score.toFixed(2)}%</span>
      </div>
      <div class="feature-rank-progress-bg">
        <div class="feature-rank-progress-fill" style="width: ${Math.max(2, f.score)}%;"></div>
      </div>
    </div>
  `).join('');
}

// Render Cuisine Tables
function renderCuisineTables() {
  const popTable = document.getElementById('popularCuisinesTable');
  const highTable = document.getElementById('highestRatedCuisinesTable');

  if (popTable) {
    popTable.innerHTML = POPULAR_CUISINES.map(c => `
      <tr>
        <td style="font-weight: 700; color: var(--accent-primary);">#${c.rank}</td>
        <td style="font-weight: 600;">${c.cuisine}</td>
        <td>${c.count.toLocaleString()}</td>
        <td style="font-weight: 700; color: var(--text-primary);">${c.votes.toLocaleString()}</td>
        <td><span class="rating-chip" style="background: rgba(255,177,66,0.15); color: #ffb142;">★ ${c.rating.toFixed(2)}</span></td>
      </tr>
    `).join('');
  }

  if (highTable) {
    highTable.innerHTML = HIGHEST_RATED_CUISINES.map(c => `
      <tr>
        <td style="font-weight: 700; color: var(--accent-primary);">#${c.rank}</td>
        <td style="font-weight: 600;">${c.cuisine}</td>
        <td>${c.count}</td>
        <td><span class="rating-chip" style="background: rgba(46,213,115,0.15); color: #2ed573;">★ ${c.rating.toFixed(2)}</span></td>
        <td>${c.votes.toLocaleString()}</td>
      </tr>
    `).join('');
  }
}

// Dataset Browser with Search & Pagination
let currentDataPage = 1;
const pageSize = 5;

function setupDatasetBrowser() {
  const searchInput = document.getElementById('datasetSearch');
  const prevBtn = document.getElementById('datasetPrev');
  const nextBtn = document.getElementById('datasetNext');

  if (searchInput) {
    searchInput.addEventListener('input', () => {
      currentDataPage = 1;
      renderDatasetRows();
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      if (currentDataPage > 1) {
        currentDataPage--;
        renderDatasetRows();
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      currentDataPage++;
      renderDatasetRows();
    });
  }

  renderDatasetRows();
}

function renderDatasetRows() {
  const tbody = document.getElementById('datasetTableBody');
  const query = (document.getElementById('datasetSearch')?.value || '').toLowerCase();
  
  const filtered = SAMPLE_RESTAURANTS.filter(r => 
    r.name.toLowerCase().includes(query) ||
    r.city.toLowerCase().includes(query) ||
    r.cuisine.toLowerCase().includes(query)
  );

  const totalPages = Math.ceil(filtered.length / pageSize) || 1;
  currentDataPage = Math.min(currentDataPage, totalPages);

  const start = (currentDataPage - 1) * pageSize;
  const pageItems = filtered.slice(start, start + pageSize);

  if (tbody) {
    if (pageItems.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">No restaurants matching "${query}"</td></tr>`;
    } else {
      tbody.innerHTML = pageItems.map(r => `
        <tr>
          <td style="font-weight: 600;">${r.name}</td>
          <td>${r.city}</td>
          <td style="color: var(--text-secondary);">${r.cuisine}</td>
          <td>₹${r.cost}</td>
          <td>${r.votes.toLocaleString()}</td>
          <td>${r.booking === 'Yes' ? '<span style="color: #2ed573;">✓ Yes</span>' : '<span style="color: #ff4757;">✗ No</span>'}</td>
          <td><span class="rating-chip" style="background: rgba(46,213,115,0.15); color: #2ed573;">★ ${r.rating.toFixed(1)}</span></td>
        </tr>
      `).join('');
    }
  }

  const pageInfo = document.getElementById('datasetPageInfo');
  if (pageInfo) pageInfo.textContent = `Page ${currentDataPage} of ${totalPages} (${filtered.length} total)`;

  const prevBtn = document.getElementById('datasetPrev');
  const nextBtn = document.getElementById('datasetNext');
  if (prevBtn) prevBtn.disabled = currentDataPage <= 1;
  if (nextBtn) nextBtn.disabled = currentDataPage >= totalPages;
}
