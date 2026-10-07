// Frontend logic: theme toggle, form submit, rendering the result.
(function () {
  var root = document.documentElement;
  var themeLabel = document.getElementById("themeLabel");

  function applyLabel() {
    themeLabel.textContent = root.getAttribute("data-theme") === "dark" ? "DAYLIGHT MODE" : "NEON MODE";
  }
  document.getElementById("themeToggle").addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    applyLabel();
  });
  applyLabel();

  var slider = document.getElementById("games");
  var sliderValue = document.getElementById("gamesValue");
  slider.addEventListener("input", function () { sliderValue.textContent = slider.value + " GAMES"; });

  var form = document.getElementById("analyzeForm");
  var errorBox = document.getElementById("error");
  var loading = document.getElementById("loading");
  var result = document.getElementById("result");
  var button = document.getElementById("submitBtn");

  // green for low, yellow/orange for moderate, bright red for high
  function colorFor(score) {
    if (score < 30) return "var(--good)";
    if (score < 55) return "var(--accent-yellow)";
    return "var(--bad)";
  }

  function render(data) {
    var color = colorFor(data.suspicion);
    document.getElementById("playerName").textContent = "@" + data.username;
    document.getElementById("scoreText").textContent = data.suspicion + "%";
    var badge = document.getElementById("verdict");
    badge.textContent = data.verdict;
    badge.style.background = color;
    badge.style.color = "#000000";
    var gauge = document.getElementById("gauge");
    gauge.style.setProperty("--pct", data.suspicion);
    gauge.style.setProperty("--color", color);
    document.getElementById("interval").textContent =
      "95% CONFIDENCE INTERVAL: " + data.ci_low + "% TO " + data.ci_high + "%";
    document.getElementById("statScored").textContent = data.games_scored;
    document.getElementById("statRating").textContent = data.avg_rating;
    document.getElementById("statFlagged").textContent = data.flagged_games;

    var body = document.getElementById("gamesBody");
    body.textContent = "";
    data.games.forEach(function (g) {
      var tr = document.createElement("tr");
      function cell(text) { var td = document.createElement("td"); td.textContent = text; tr.appendChild(td); return td; }
      cell(g.date);
      var opp = document.createElement("td");
      var a = document.createElement("a");
      a.textContent = g.opponent;
      a.href = g.url; a.target = "_blank"; a.rel = "noopener";
      opp.appendChild(a); tr.appendChild(opp);
      cell((g.time_class || "standard").toUpperCase());
      cell(g.rating);
      var td = document.createElement("td");
      var bar = document.createElement("div"); bar.className = "bar";
      var track = document.createElement("div"); track.className = "bar-track";
      var fill = document.createElement("div"); fill.className = "bar-fill";
      fill.style.width = g.probability + "%"; fill.style.background = colorFor(g.probability);
      track.appendChild(fill);
      var num = document.createElement("span"); num.textContent = g.probability + "%";
      bar.appendChild(track); bar.appendChild(num); td.appendChild(bar); tr.appendChild(td);
      body.appendChild(tr);
    });
    result.hidden = false;
  }

  var progressTimer = null;
  var progressBar = document.getElementById("progressBar");
  var loadingPercent = document.getElementById("loadingPercent");

  function setStepActive(i) {
    var s = document.getElementById("step" + i);
    if (s) s.className = "step-item active";
  }

  function setStepCompleted(i) {
    var s = document.getElementById("step" + i);
    if (s) s.className = "step-item completed";
  }

  function resetSteps() {
    [1, 2, 3, 4].forEach(function (i) {
      var s = document.getElementById("step" + i);
      if (s) s.className = i === 1 ? "step-item active" : "step-item";
    });
    if (progressBar) progressBar.style.width = "0%";
    if (loadingPercent) loadingPercent.textContent = "0%";
  }

  function startProgressAnimation() {
    resetSteps();
    var currentPct = 0;
    var stepIndex = 1;
    if (progressTimer) clearInterval(progressTimer);

    progressTimer = setInterval(function () {
      if (currentPct < 95) {
        currentPct += Math.floor(Math.random() * 5) + 3;
        if (currentPct > 95) currentPct = 95;
        if (progressBar) progressBar.style.width = currentPct + "%";
        if (loadingPercent) loadingPercent.textContent = currentPct + "%";

        if (currentPct >= 25 && stepIndex === 1) {
          setStepCompleted(1);
          setStepActive(2);
          stepIndex = 2;
        } else if (currentPct >= 60 && stepIndex === 2) {
          setStepCompleted(2);
          setStepActive(3);
          stepIndex = 3;
        } else if (currentPct >= 85 && stepIndex === 3) {
          setStepCompleted(3);
          setStepActive(4);
          stepIndex = 4;
        }
      }
    }, 200);
  }

  function finishProgressAnimation(callback) {
    if (progressTimer) clearInterval(progressTimer);
    if (progressBar) progressBar.style.width = "100%";
    if (loadingPercent) loadingPercent.textContent = "100%";
    [1, 2, 3, 4].forEach(setStepCompleted);
    setTimeout(callback, 350);
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    errorBox.hidden = true; result.hidden = true; loading.hidden = false; button.disabled = true;
    startProgressAnimation();

    fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: document.getElementById("username").value, games: slider.value })
    })
      .then(function (res) { return res.json().then(function (data) { return { ok: res.ok, data: data }; }); })
      .then(function (r) {
        if (!r.ok) throw new Error(r.data.error || "Something went wrong.");
        finishProgressAnimation(function () {
          loading.hidden = true;
          render(r.data);
          button.disabled = false;
        });
      })
      .catch(function (err) {
        if (progressTimer) clearInterval(progressTimer);
        loading.hidden = true;
        errorBox.textContent = err.message;
        errorBox.hidden = false;
        button.disabled = false;
      });
  });
})();


