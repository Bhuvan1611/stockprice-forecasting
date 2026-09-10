(function () {
  "use strict";

  // ---------- Theme toggle ----------
  var root = document.documentElement;
  var saved = localStorage.getItem("ledger-theme");
  if (saved) root.setAttribute("data-theme", saved);

  function currentTheme() {
    return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }

  function setToggleLabel(btn) {
    btn.textContent = currentTheme() === "dark" ? "Light mode" : "Dark mode";
  }

  document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
    setToggleLabel(btn);
    btn.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("ledger-theme", next);
      setToggleLabel(btn);
    });
  });

  // ---------- Ticker chips: fill nearest ticker input ----------
  document.querySelectorAll("[data-chip]").forEach(function (chip) {
    chip.addEventListener("click", function () {
      var targetSel = chip.getAttribute("data-target") || "input[name='stock']";
      var target = document.querySelector(targetSel);
      if (target) {
        target.value = chip.getAttribute("data-chip");
        target.focus();
      }
    });
  });

  // ---------- Loading overlay on form submit ----------
  var overlay = document.getElementById("loading-overlay");
  document.querySelectorAll("form[data-loading]").forEach(function (form) {
    form.addEventListener("submit", function () {
      if (overlay) overlay.classList.add("active");
      var btn = form.querySelector("button[type='submit']");
      if (btn) btn.setAttribute("disabled", "disabled");
    });
  });

  // ---------- Ticker tape ----------
  var tapeEl = document.getElementById("ticker-tape");
  if (tapeEl) {
    fetch("/api/ticker-tape")
      .then(function (r) { return r.json(); })
      .then(function (quotes) {
        if (!quotes || !quotes.length) {
          tapeEl.closest(".tape-wrap").style.display = "none";
          return;
        }
        var html = quotes.map(function (q) {
          var cls = q.up ? "up" : "down";
          var arrow = q.up ? "▲" : "▼";
          return '<span class="sym">' + q.symbol + '</span>' +
                 q.price.toLocaleString() +
                 ' <span class="' + cls + '">' + arrow + ' ' + Math.abs(q.change_pct) + '%</span>';
        }).join("");
        // duplicate content so the CSS scroll loop is seamless
        tapeEl.innerHTML = html + html;
      })
      .catch(function () {
        tapeEl.closest(".tape-wrap").style.display = "none";
      });
  }

  // ---------- Watchlist add/remove via fetch (used on index + watchlist pages) ----------
  document.querySelectorAll("[data-watchlist-add]").forEach(function (el) {
    el.addEventListener("click", function (evt) {
      evt.preventDefault();
      var ticker = el.getAttribute("data-watchlist-add");
      var fd = new FormData();
      fd.append("stock", ticker);
      fetch("/watchlist/add", { method: "POST", body: fd })
        .then(function () {
          el.textContent = "Added";
          el.setAttribute("disabled", "disabled");
        });
    });
  });

  document.querySelectorAll("[data-watchlist-remove]").forEach(function (el) {
    el.addEventListener("click", function (evt) {
      evt.preventDefault();
      var ticker = el.getAttribute("data-watchlist-remove");
      var row = el.closest(".watch-row");
      fetch("/watchlist/remove/" + encodeURIComponent(ticker), { method: "POST" })
        .then(function () {
          if (row) row.remove();
        });
    });
  });
})();
