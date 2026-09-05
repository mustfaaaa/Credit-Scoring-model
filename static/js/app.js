/* Credit Scoring Dashboard — tab navigation, presets and live scoring.
   Vanilla JS: no framework, no build step, works offline. */
(function () {
  "use strict";

  /* ------------------------------------------------------------- tabs --- */
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const panels = tabs.map((t) => document.getElementById(t.getAttribute("aria-controls")));

  function selectTab(index, focus) {
    tabs.forEach((tab, i) => {
      const selected = i === index;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      panels[i].hidden = !selected;
    });
    if (focus) tabs[index].focus();
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(i, false));
    // Roving tabindex: arrow keys move between tabs, per the ARIA pattern.
    tab.addEventListener("keydown", (event) => {
      const map = { ArrowRight: 1, ArrowLeft: -1, Home: "first", End: "last" };
      const action = map[event.key];
      if (action === undefined) return;
      event.preventDefault();
      if (action === "first") return selectTab(0, true);
      if (action === "last") return selectTab(tabs.length - 1, true);
      selectTab((i + action + tabs.length) % tabs.length, true);
    });
  });
  selectTab(0, false);

  /* ---------------------------------------------------------- scoring --- */
  const form = document.getElementById("score-form");
  if (!form) return;

  const button = document.getElementById("score-btn");
  const buttonText = button.querySelector(".btn-text");
  const resultBox = document.getElementById("result");
  const emptyBox = document.getElementById("result-empty");
  const summary = document.getElementById("error-summary");
  const summaryList = document.getElementById("error-list");

  const PRESETS = {
    strong: {
      LIMIT_BAL: 350000, SEX: "female", EDUCATION: "graduate_school",
      MARRIAGE: "married", AGE: 41,
      PAY_0: -1, PAY_2: -1, PAY_3: -1, PAY_4: -1, PAY_5: -1, PAY_6: -1,
      BILL_AMT1: 28000, BILL_AMT2: 26500, BILL_AMT3: 31000,
      BILL_AMT4: 24000, BILL_AMT5: 22500, BILL_AMT6: 20000,
      PAY_AMT1: 26500, PAY_AMT2: 31000, PAY_AMT3: 24000,
      PAY_AMT4: 22500, PAY_AMT5: 20000, PAY_AMT6: 19000
    },
    distressed: {
      LIMIT_BAL: 20000, SEX: "male", EDUCATION: "high_school",
      MARRIAGE: "single", AGE: 24,
      PAY_0: 3, PAY_2: 3, PAY_3: 2, PAY_4: 2, PAY_5: 1, PAY_6: 1,
      BILL_AMT1: 19800, BILL_AMT2: 19400, BILL_AMT3: 19100,
      BILL_AMT4: 18600, BILL_AMT5: 18000, BILL_AMT6: 17500,
      PAY_AMT1: 0, PAY_AMT2: 700, PAY_AMT3: 800,
      PAY_AMT4: 0, PAY_AMT5: 600, PAY_AMT6: 700
    }
  };

  function fill(values) {
    Object.entries(values).forEach(([name, value]) => {
      const field = form.elements[name];
      if (!field) return;
      field.value = value;
      clearFieldError(name);
    });
  }

  function clearFieldError(name) {
    const field = form.elements[name];
    const slot = document.getElementById(name + "-error");
    if (field) field.removeAttribute("aria-invalid");
    if (slot) slot.textContent = "";
  }

  function clearAllErrors() {
    summary.hidden = true;
    summaryList.innerHTML = "";
    Array.from(form.elements).forEach((el) => el.name && clearFieldError(el.name));
  }

  /* Errors appear inline next to each field AND in a focusable summary that
     links back to them — inline-only or summary-only both fail accessibility. */
  function showErrors(errors) {
    summaryList.innerHTML = "";
    errors.forEach(({ field, message }) => {
      const input = form.elements[field];
      const slot = document.getElementById(field + "-error");
      if (input) input.setAttribute("aria-invalid", "true");
      if (slot) slot.textContent = message;

      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = "#" + field;
      link.textContent = field + ": " + message;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        if (input) input.focus();
      });
      item.appendChild(link);
      summaryList.appendChild(item);
    });
    summary.hidden = false;
    summary.focus();
  }

  function collect() {
    const payload = {};
    Array.from(form.elements).forEach((el) => {
      if (el.name) payload[el.name] = el.value;
    });
    return payload;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
  }

  function riskClass(category) {
    if (category.startsWith("LOW")) return "low";
    if (category.startsWith("MODERATE")) return "moderate";
    if (category.startsWith("VERY")) return "very";
    return "high";
  }

  function render(data, actualOutcome) {
    const approved = data.decision === "APPROVE";
    const threshold = window.THRESHOLD || 0.5;

    const indicators = (data.indicators || []).map((ind) => `
      <tr>
        <td>${escapeHtml(ind.label)}</td>
        <td class="${ind.flagged ? "indicator-flag" : ""}">${escapeHtml(ind.value)}</td>
        <td>${escapeHtml(ind.avg_repaid || "—")}</td>
        <td>${escapeHtml(ind.avg_defaulted || "—")}</td>
      </tr>`).join("");

    const actualNote = actualOutcome ? `
      <div class="note" style="margin-top:var(--space-4)">
        <strong>Ground truth:</strong> this is a real customer from the dataset who
        actually <strong>${escapeHtml(actualOutcome)}</strong>.
      </div>` : "";

    resultBox.innerHTML = `
      <div class="verdict ${approved ? "approve" : "decline"}">
        <div class="decision">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
               style="width:24px;height:24px" aria-hidden="true">
            ${approved ? '<path d="m5 12 5 5L20 7"/>'
                       : '<circle cx="12" cy="12" r="9"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>'}
          </svg>
          ${escapeHtml(data.decision)}
        </div>
        <div class="sub">${escapeHtml(data.prediction)}</div>
      </div>

      <div style="margin-bottom:var(--space-4)">
        <div class="prob-bar">
          <div class="prob-fill" style="width:${data.percent_default}%">
            ${data.percent_default}%
          </div>
          <div class="threshold-marker" style="left:${(threshold * 100).toFixed(1)}%"></div>
        </div>
        <div class="threshold-label">
          Probability of default · amber line = decision threshold ${threshold.toFixed(3)}
        </div>
      </div>

      <div class="grid grid-kpi" style="margin-bottom:var(--space-4)">
        <div class="kpi bad">
          <span class="label">Bad credit</span>
          <span class="value">${data.percent_default}%</span>
          <span class="note">probability of default</span>
        </div>
        <div class="kpi good">
          <span class="label">Good credit</span>
          <span class="value">${data.percent_good}%</span>
          <span class="note">probability of repayment</span>
        </div>
        <div class="kpi accent">
          <span class="label">Risk band</span>
          <span class="value" style="font-size:1rem">
            <span class="chip ${riskClass(data.risk_category)}">${escapeHtml(data.risk_category)}</span>
          </span>
          <span class="note">model: ${escapeHtml(data.model)}</span>
        </div>
      </div>

      <h3 style="font-size:0.95rem;margin-bottom:var(--space-2)">Why &mdash; this applicant&rsquo;s risk indicators</h3>
      <p class="hint">
        The actual engineered values the model saw, next to the dataset averages.
        <span class="indicator-flag">Red</span> means the applicant is at or beyond the
        typical defaulter on that measure.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Indicator</th>
              <th scope="col">This applicant</th>
              <th scope="col">Avg. repaid</th>
              <th scope="col">Avg. defaulted</th>
            </tr>
          </thead>
          <tbody>${indicators}</tbody>
        </table>
      </div>
      ${actualNote}`;

    emptyBox.hidden = true;
    resultBox.hidden = false;
  }

  let pendingOutcome = null;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearAllErrors();
    button.disabled = true;
    buttonText.textContent = "Scoring…";

    try {
      const response = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collect())
      });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        showErrors(data.errors || [{ field: "LIMIT_BAL", message: "Scoring failed." }]);
      } else {
        render(data, pendingOutcome);
        pendingOutcome = null;
      }
    } catch (err) {
      showErrors([{ field: "LIMIT_BAL", message: "Could not reach the server: " + err.message }]);
    } finally {
      button.disabled = false;
      buttonText.textContent = "Score this applicant";
    }
  });

  // Clear a field's error as soon as the user edits it.
  form.addEventListener("input", (event) => {
    if (event.target.name) clearFieldError(event.target.name);
  });

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      clearAllErrors();
      pendingOutcome = null;
      fill(PRESETS[btn.dataset.preset]);
    });
  });

  document.querySelectorAll("[data-sample]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const original = btn.textContent;
      btn.disabled = true;
      btn.textContent = "Loading…";
      try {
        const response = await fetch("/api/sample?kind=" + encodeURIComponent(btn.dataset.sample));
        const data = await response.json();
        clearAllErrors();
        fill(data.applicant);
        pendingOutcome = data.actual_outcome;
      } catch (err) {
        showErrors([{ field: "LIMIT_BAL", message: "Could not load a sample: " + err.message }]);
      } finally {
        btn.disabled = false;
        btn.textContent = original;
      }
    });
  });
})();
