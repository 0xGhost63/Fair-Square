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
  function syncSlider() {
    sliderValue.textContent = slider.value + " GAMES";
  }
  slider.addEventListener("input", syncSlider);
  slider.addEventListener("change", syncSlider);
  syncSlider();

  var depthSlider = document.getElementById("depth");
  var depthValue = document.getElementById("depthValue");
  function syncDepth() {
    if (!depthSlider || !depthValue) return;
    var d = parseInt(depthSlider.value, 10);
    var label = "DEPTH " + d;
    if (d === 10) label += " (OPTIMAL)";
    else if (d <= 8) label += " (FAST)";
    else if (d >= 12) label += " (DEEP)";
    depthValue.textContent = label;
  }
  if (depthSlider) {
    depthSlider.addEventListener("input", syncDepth);
    depthSlider.addEventListener("change", syncDepth);
    syncDepth();
  }

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
    var intervalText = "95% CONFIDENCE INTERVAL: " + data.ci_low + "% TO " + data.ci_high + "%";
    if (data.games_requested && data.games_scored < data.games_requested) {
      intervalText = "[ALL " + data.games_scored + " AVAILABLE RATED GAMES ANALYZED]  " + intervalText;
    }
    document.getElementById("interval").textContent = intervalText;
    
    var scoredText = String(data.games_scored);
    if (data.games_requested && data.games_scored < data.games_requested) {
      scoredText = data.games_scored + " / " + data.games_requested;
    }
    document.getElementById("statScored").textContent = scoredText;
    document.getElementById("statRating").textContent = data.avg_rating;
    document.getElementById("statFlagged").textContent = data.flagged_games;
    document.getElementById("statAcpl").textContent = data.avg_acpl != null ? data.avg_acpl + " CP" : "N/A";
    document.getElementById("statTop1").textContent = data.avg_top1 != null ? data.avg_top1 + "%" : "N/A";
    document.getElementById("statBlunder").textContent = data.avg_blunder_rate != null ? data.avg_blunder_rate + "%" : "N/A";

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
      cell(g.acpl != null ? g.acpl + " cp" : "-");
      cell(g.top1_match != null ? g.top1_match + "%" : "-");
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
  var step3Counter = document.getElementById("step3Counter");
  var step4Counter = document.getElementById("step4Counter");
  var step3Text = document.getElementById("step3Text");
  var step4Text = document.getElementById("step4Text");

  function setStepActive(i) {
    var s = document.getElementById("step" + i);
    if (s) s.className = "step-item active";
  }

  function setStepCompleted(i) {
    var s = document.getElementById("step" + i);
    if (s) s.className = "step-item completed";
  }

  function resetSteps(totalGames, depthVal) {
    [1, 2, 3, 4].forEach(function (i) {
      var s = document.getElementById("step" + i);
      if (s) s.className = i === 1 ? "step-item active" : "step-item";
    });
    if (progressBar) progressBar.style.width = "0%";
    if (loadingPercent) loadingPercent.textContent = "0%";
    if (step3Counter) { step3Counter.textContent = "0 / " + totalGames; step3Counter.hidden = false; }
    if (step4Counter) { step4Counter.textContent = "0 / " + totalGames; step4Counter.hidden = true; }
    if (step3Text) step3Text.textContent = "Evaluating move accuracy with Stockfish 16 Engine (Depth " + (depthVal || 10) + ")...";
    if (step4Text) step4Text.textContent = "Evaluating Hybrid Engine + Timing Model...";
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

    var username = document.getElementById("username").value.trim();
    var gamesCount = parseInt(slider.value, 10);
    var depthVal = depthSlider ? parseInt(depthSlider.value, 10) : 10;
    resetSteps(gamesCount, depthVal);

    if (window.EventSource) {
      var sseUrl = "/api/analyze/stream?username=" + encodeURIComponent(username) + "&games=" + gamesCount + "&depth=" + depthVal;
      var es = new EventSource(sseUrl);
      var receivedAnyEvent = false;

      es.onmessage = function (event) {
        receivedAnyEvent = true;
        try {
          var data = JSON.parse(event.data);
          if (data.type === "step") {
            if (data.step === 1) { setStepActive(1); }
            else if (data.step === 2) { setStepCompleted(1); setStepActive(2); }
            else if (data.step === 3) { setStepCompleted(2); setStepActive(3); }
            else if (data.step === 4) { setStepCompleted(3); setStepActive(4); }
          } else if (data.type === "progress") {
            var cur = data.current;
            var tot = data.total;
            if (step3Counter) {
              step3Counter.textContent = cur + " / " + tot;
              step3Counter.hidden = false;
            }
            if (step4Counter && cur >= tot) {
              step4Counter.textContent = cur + " / " + tot;
              step4Counter.hidden = false;
            }
            if (data.opponent && step3Text) {
              step3Text.textContent = "Evaluating moves vs @" + data.opponent + " (" + cur + "/" + tot + ")...";
            }
            var pct = Math.min(95, Math.max(10, Math.round((cur / tot) * 95)));
            if (progressBar) progressBar.style.width = pct + "%";
            if (loadingPercent) loadingPercent.textContent = pct + "%";

            if (cur >= tot) {
              setStepCompleted(3);
              setStepActive(4);
            } else {
              setStepCompleted(2);
              setStepActive(3);
            }
          } else if (data.type === "done") {
            es.close();
            finishProgressAnimation(function () {
              loading.hidden = true;
              render(data.result);
              button.disabled = false;
            });
          } else if (data.type === "error") {
            es.close();
            loading.hidden = true;
            errorBox.textContent = data.error || "Analysis error occurred.";
            errorBox.hidden = false;
            button.disabled = false;
          }
        } catch (err) {
          console.error("SSE parse error", err);
        }
      };

      es.onerror = function () {
        es.close();
        // If SSE failed before getting events, fallback to POST /api/analyze
        if (!receivedAnyEvent) {
          fallbackPost(username, gamesCount, depthVal);
        } else {
          loading.hidden = true;
          errorBox.textContent = "Connection interrupted during analysis. Please try again.";
          errorBox.hidden = false;
          button.disabled = false;
        }
      };
    } else {
      fallbackPost(username, gamesCount, depthVal);
    }

    function fallbackPost(u, g, d) {
      fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: u, games: g, depth: d })
      })
        .then(function (res) { return res.json().then(function (d) { return { ok: res.ok, data: d }; }); })
        .then(function (r) {
          if (!r.ok) throw new Error(r.data.error || "Something went wrong.");
          finishProgressAnimation(function () {
            loading.hidden = true;
            render(r.data);
            button.disabled = false;
          });
        })
        .catch(function (err) {
          loading.hidden = true;
          errorBox.textContent = err.message;
          errorBox.hidden = false;
          button.disabled = false;
        });
    }
  });
})();


