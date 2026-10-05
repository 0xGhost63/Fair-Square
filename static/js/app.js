let currentTaskId = null;
let pollTimer = null;
let extractedPgnPlayers = { white: "White Player", black: "Black Player" };

function initTheme() {
  const savedTheme = localStorage.getItem('fair_square_theme') || 'light';
  applyTheme(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  applyTheme(newTheme);
  localStorage.setItem('fair_square_theme', newTheme);
}

function applyTheme(theme) {
  if (theme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.getElementById('theme-toggle-btn').innerText = 'Theme: Dark';
  } else {
    document.documentElement.removeAttribute('data-theme');
    document.getElementById('theme-toggle-btn').innerText = 'Theme: Light';
  }
}

document.addEventListener('DOMContentLoaded', initTheme);

function switchTab(tabName) {
  document.getElementById('tab-chesscom').classList.remove('active');
  document.getElementById('tab-pgn').classList.remove('active');
  document.getElementById('form-chesscom').style.display = 'none';
  document.getElementById('form-pgn').style.display = 'none';

  if (tabName === 'chesscom') {
    document.getElementById('tab-chesscom').classList.add('active');
    document.getElementById('form-chesscom').style.display = 'block';
  } else {
    document.getElementById('tab-pgn').classList.add('active');
    document.getElementById('form-pgn').style.display = 'block';
  }
}

async function autoExtractPgnPlayers() {
  const pgnText = document.getElementById('pgn_text').value.trim();
  if (!pgnText) return;

  try {
    const resp = await fetch('/api/parse-pgn-players', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pgn_text: pgnText })
    });
    const data = await resp.json();
    if (data.status === 'success' && data.players) {
      extractedPgnPlayers = data.players;
      const selectElem = document.getElementById('pgn_selected_player');
      selectElem.innerHTML = `
        <option value="white" selected>White: ${data.players.white}</option>
        <option value="black">Black: ${data.players.black}</option>
      `;
    }
  } catch (err) {
    console.error('Player extraction error:', err);
  }
}

async function submitChessComSearch(e) {
  e.preventDefault();
  const username = document.getElementById('username').value.trim();
  const maxGames = document.getElementById('max_games').value;
  const depth = document.getElementById('engine_depth').value;

  if (!username) return;

  showStatus('Connecting to Chess.com API and launching analysis...', 10);
  document.getElementById('results-section').style.display = 'none';

  try {
    const resp = await fetch('/api/analyze/chesscom', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: username,
        max_games: parseInt(maxGames),
        depth: parseInt(depth)
      })
    });

    const data = await resp.json();
    if (data.status === 'started') {
      currentTaskId = data.task_id;
      document.getElementById('task-id-display').innerText = currentTaskId;
      startPollingTask(currentTaskId);
    } else {
      showError(data.message || 'Failed to start analysis.');
    }
  } catch (err) {
    showError('Error initiating analysis: ' + err.message);
  }
}

async function submitPgnUpload(e) {
  e.preventDefault();
  const pgnText = document.getElementById('pgn_text').value.trim();
  const targetColor = document.getElementById('pgn_selected_player').value;
  const playerName = targetColor === 'white' ? extractedPgnPlayers.white : extractedPgnPlayers.black;
  const depth = document.getElementById('pgn_engine_depth').value;

  if (!pgnText) return;

  showStatus(`Parsing PGN games for ${playerName} (${targetColor.toUpperCase()}) and launching Stockfish...`, 10);
  document.getElementById('results-section').style.display = 'none';

  try {
    const resp = await fetch('/api/analyze/pgn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pgn_text: pgnText,
        target_color: targetColor,
        player_name: playerName,
        depth: parseInt(depth)
      })
    });

    const data = await resp.json();
    if (data.status === 'started') {
      currentTaskId = data.task_id;
      document.getElementById('task-id-display').innerText = currentTaskId;
      startPollingTask(currentTaskId);
    } else {
      showError(data.message || 'Failed to start PGN analysis.');
    }
  } catch (err) {
    showError('Error initiating PGN analysis: ' + err.message);
  }
}

function showStatus(msg, pct) {
  document.getElementById('status-section').style.display = 'block';
  document.getElementById('status-message').innerText = msg;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-bar').style.background = 'var(--cyan-accent)';
}

function showError(msg) {
  document.getElementById('status-section').style.display = 'block';
  document.getElementById('status-message').innerText = 'ERROR: ' + msg;
  document.getElementById('progress-bar').style.width = '100%';
  document.getElementById('progress-bar').style.background = '#FF5757';
}

function startPollingTask(taskId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/task/${taskId}`);
      const data = await resp.json();

      if (data.status === 'running') {
        showStatus(data.progress_message || 'Analyzing move positions...', data.progress_pct || 50);
      } else if (data.status === 'completed') {
        clearInterval(pollTimer);
        document.getElementById('status-section').style.display = 'none';
        renderResults(data.result);
      } else if (data.status === 'failed') {
        clearInterval(pollTimer);
        showError(data.error || 'Analysis task failed.');
      }
    } catch (err) {
      console.error('Polling error:', err);
    }
  }, 2000);
}

function renderResults(report) {
  document.getElementById('results-section').style.display = 'block';
  document.getElementById('report-player-name').innerText = report.player_name.toUpperCase();
  document.getElementById('report-risk-badge').innerText = report.risk_level;

  const scoreVal = report.suspicion_score_pct;
  const scoreElem = document.getElementById('suspicion-score-val');
  scoreElem.innerText = scoreVal + '%';

  const scoreCard = document.getElementById('score-card');
  scoreCard.className = 'score-box';
  if (scoreVal >= 75) {
    scoreCard.classList.add('high');
  } else if (scoreVal >= 45) {
    scoreCard.classList.add('moderate');
  } else {
    scoreCard.classList.add('low');
  }

  document.getElementById('risk-level-val').innerText = report.risk_level;
  document.getElementById('res-total-games').innerText = report.total_games_analyzed;
  document.getElementById('res-player-elo').innerText = report.player_elo;
  document.getElementById('res-confidence').innerText = report.confidence_pct + '%';

  const factorsContainer = document.getElementById('factors-container');
  factorsContainer.innerHTML = '';
  if (report.top_contributing_factors && report.top_contributing_factors.length > 0) {
    report.top_contributing_factors.forEach(f => {
      const div = document.createElement('div');
      div.style.cssText = 'border: 3px solid var(--border-color); padding: 12px; margin-bottom: 10px; background: var(--card-bg); box-shadow: 3px 3px 0px var(--shadow-color);';
      div.innerHTML = `<strong>${f.feature_label}</strong><br>${f.description}`;
      factorsContainer.appendChild(div);
    });
  } else {
    factorsContainer.innerHTML = '<p style="font-family: var(--font-mono);">All metrics fall within expected human rating variance.</p>';
  }

  const tbody = document.getElementById('games-table-body');
  tbody.innerHTML = '';
  if (report.game_breakdown) {
    report.game_breakdown.forEach(g => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${g.event}</td>
        <td>${g.white} vs ${g.black}</td>
        <td>${g.moves_analyzed}</td>
        <td>${g.avg_cpl} cp</td>
        <td>${g.sf_match_pct}%</td>
        <td>${g.max_streak} moves</td>
        <td>${g.streaks_count}</td>
        <td>${g.critical_match_pct}%</td>
        <td><strong>${g.suspicion_prob}%</strong></td>
      `;
      tbody.appendChild(tr);
    });
  }

  document.getElementById('res-legal-notice').innerText = report.legal_notice;
}
