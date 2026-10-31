// Draws the form from /api/schema, sends a request, shows only the result, and clears everything entered.
// Nothing is kept in the browser: no storage, no cookies, and the entries are wiped after each submission.
(async function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  function el(tag, attributes, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attributes || {})) {
      if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
    for (const child of children) node.append(child);
    return node;
  }

  const GROUPS = [
    { title: "The stay", numeric: true },
    { title: "Patient", choices: ["race", "gender", "age", "weight"] },
    { title: "Admission and discharge",
      choices: ["admission_type_id", "discharge_disposition_id", "admission_source_id", "payer_code", "medical_specialty"] },
    { title: "Diagnoses", codes: true },
    { title: "Tests", choices: ["max_glu_serum", "A1Cresult"] },
    { title: "Diabetes treatment", choices: ["change", "diabetesMed"] },
  ];

  let schema;
  try {
    const response = await fetch("/api/schema", { cache: "no-store" });
    schema = await response.json();
  } catch (error) {
    $("form-panel").prepend(el("p", { class: "errors", text: "The page could not load its settings. Please reload." }));
    return;
  }

  // ---- form -------------------------------------------------------------------------------------------

  function fieldWrap(name, label, control, hint) {
    const wrap = el("div", { class: "field" }, el("label", { for: `f-${name}`, text: label }), control);
    if (hint) wrap.append(el("span", { class: "hint", text: hint }));
    return wrap;
  }

  function selectControl(name, options) {
    const select = el("select", { id: `f-${name}`, name, required: "" });
    select.append(el("option", { value: "", text: "Select..." }));
    for (const option of options) select.append(el("option", { value: option.value, text: option.label }));
    return select;
  }

  function buildForm() {
    const container = $("form-fields");
    const byName = (list) => Object.fromEntries(list.map((item) => [item.name, item]));
    const choices = byName(schema.choices);
    for (const group of GROUPS) {
      const grid = el("div", { class: "grid" });
      if (group.numeric) {
        for (const field of schema.numeric) {
          const input = el("input", { id: `f-${field.name}`, name: field.name, type: "number", step: "1",
                                      min: field.min, max: field.max, required: "", inputmode: "numeric" });
          grid.append(fieldWrap(field.name, field.label, input, `${field.hint} Whole number, ${field.min} to ${field.max}.`));
        }
      }
      for (const name of group.choices || []) {
        grid.append(fieldWrap(name, choices[name].label, selectControl(name, choices[name].options), choices[name].hint));
      }
      if (group.codes) {
        for (const field of schema.codes) {
          const attributes = { id: `f-${field.name}`, name: field.name, type: "text", maxlength: "8",
                               placeholder: field.optional ? "optional" : "e.g. 250.83", autocapitalize: "characters" };
          if (!field.optional) attributes.required = "";
          const label = field.optional ? `${field.label} (optional)` : field.label;
          grid.append(fieldWrap(field.name, label, el("input", attributes), field.hint));
        }
      }
      const titleId = `section-${container.children.length}`;
      container.append(el("div", { class: "section", role: "group", "aria-labelledby": titleId },
        el("h3", { id: titleId, text: group.title }), grid));
    }
  }

  function addMedicationRow(drug, level) {
    const drugSelect = el("select", { "aria-label": "Medication" });
    drugSelect.append(el("option", { value: "", text: "Choose a medication..." }));
    for (const option of schema.medications.options) {
      drugSelect.append(el("option", { value: option.name, text: option.name }));
    }
    const levelSelect = el("select", { "aria-label": "Change in dose" });
    for (const value of ["No", "Steady", "Up", "Down"]) levelSelect.append(el("option", { value, text: value }));
    levelSelect.value = schema.medications.default;
    const remove = el("button", { type: "button", class: "secondary", "aria-label": "Remove this medication", text: "Remove" });
    const row = el("div", { class: "medication-row" },
      el("div", { class: "field" }, el("label", { text: "Medication" }), drugSelect),
      el("div", { class: "field" }, el("label", { text: "Change" }), levelSelect),
      remove);
    remove.addEventListener("click", () => row.remove());
    if (drug) { drugSelect.value = drug; levelSelect.value = level; }
    $("medication-rows").append(row);
  }

  function resetMedications() {
    $("medication-rows").replaceChildren();
    addMedicationRow();
  }

  function collectValues() {
    const values = {};
    for (const field of schema.numeric) {
      const text = $(`f-${field.name}`).value;
      values[field.name] = text === "" ? "" : Number(text);
    }
    for (const field of [...schema.choices, ...schema.codes]) values[field.name] = $(`f-${field.name}`).value;
    const medications = {};
    for (const row of $("medication-rows").children) {
      const [drug, level] = row.querySelectorAll("select");
      if (drug.value) medications[drug.value] = level.value;
    }
    values.medications = medications;
    return values;
  }

  function fillExample() {
    const example = schema.example;
    for (const field of [...schema.numeric, ...schema.choices, ...schema.codes]) {
      $(`f-${field.name}`).value = String(example[field.name]);
    }
    $("medication-rows").replaceChildren();
    for (const [drug, level] of Object.entries(example.medications)) addMedicationRow(drug, level);
  }

  function clearForm() {
    $("encounter-form").reset();
    resetMedications();
    $("file-form").reset();
    showErrors($("form-errors"), []);
    showErrors($("file-errors"), []);
  }

  // ---- sending and showing results ------------------------------------------------------------------------

  function showErrors(container, messages) {
    container.replaceChildren();
    container.hidden = messages.length === 0;
    if (!messages.length) return;
    const list = el("ul");
    for (const message of messages) list.append(el("li", { text: message }));
    container.append(list);
  }

  async function post(url, body, contentType) {
    const response = await fetch(url, { method: "POST", headers: { "Content-Type": contentType }, body, cache: "no-store" });
    let data = {};
    try { data = await response.json(); } catch (error) { /* keep the generic message */ }
    if (!response.ok) throw new Error(JSON.stringify(data.errors || ["Something went wrong. Please try again."]));
    return data;
  }

  function showResults(data) {
    const body = $("result-table").tBodies[0];
    body.replaceChildren();
    for (const result of data.results) {
      const badge = el("span", { class: `badge ${result.flagged ? "flagged" : "not-flagged"}`,
                                 text: result.flagged ? "Flagged" : "Not flagged" });
      body.append(el("tr", {},
        el("td", { text: String(result.row) }),
        el("td", { text: result.risk_score.toFixed(3) }),
        el("td", {}, badge, el("span", { class: "result-message", text: result.message }))));
    }
    $("result-notes").replaceChildren(...data.notes.map((note) => el("li", { text: note })));
    $("input-view").hidden = true;
    $("result-view").hidden = false;
    $("result-heading").focus();
  }

  // The form names fields by their labels ("Days in hospital"); uploaded files keep the CSV column names.
  const fieldLabels = Object.fromEntries(
    [...schema.numeric, ...schema.choices, ...schema.codes].map((field) => [field.name, field.label.replace(/\?$/, "")]));

  function withLabel(message) {
    const parts = /^([A-Za-z0-9_-]+): (.*)$/.exec(message);
    return parts && fieldLabels[parts[1]] ? `${fieldLabels[parts[1]]}: ${parts[2]}` : message;
  }

  async function submit(container, button, send, format = (message) => message) {
    showErrors(container, []);
    button.disabled = true;
    try {
      const data = await send();
      clearForm();
      showResults(data);
    } catch (error) {
      let messages;
      try { messages = JSON.parse(error.message); } catch (parseError) { messages = [error.message]; }
      showErrors(container, (Array.isArray(messages) ? messages : [String(messages)]).map(format));
    } finally {
      button.disabled = false;
    }
  }

  // ---- upload directions ----------------------------------------------------------------------------------

  function buildGuide() {
    const table = el("table", { class: "guide-table" },
      el("thead", {}, el("tr", {}, el("th", { text: "Column" }), el("th", { text: "Required" }),
        el("th", { text: "Meaning" }), el("th", { text: "Allowed values" }))));
    const body = el("tbody");
    for (const row of schema.guide) {
      body.append(el("tr", {}, el("td", { text: row.column }), el("td", { text: row.required ? "yes" : "no" }),
        el("td", { text: row.meaning }), el("td", { text: row.allowed })));
    }
    table.append(body);
    $("column-guide").append(table);
    $("limit-text").textContent = `At most ${schema.limits.max_rows} rows and ${Math.round(schema.limits.max_upload_bytes / 1000)} KB.`;
  }

  // ---- wiring -------------------------------------------------------------------------------------------

  function selectTab(which) {
    const isForm = which === "form";
    $("tab-form").setAttribute("aria-selected", String(isForm));
    $("tab-file").setAttribute("aria-selected", String(!isForm));
    $("form-panel").hidden = !isForm;
    $("file-panel").hidden = isForm;
  }

  buildForm();
  $("medication-hint").textContent = schema.medications.hint;
  resetMedications();
  buildGuide();
  $("model-text").textContent = `Model: ${schema.model.model}.`;

  $("tab-form").addEventListener("click", () => selectTab("form"));
  $("tab-file").addEventListener("click", () => selectTab("file"));
  $("add-medication").addEventListener("click", () => addMedicationRow());
  $("fill-example").addEventListener("click", fillExample);

  $("encounter-form").addEventListener("submit", (event) => {
    event.preventDefault();
    submit($("form-errors"), event.submitter, () =>
      post("/api/predict", JSON.stringify(collectValues()), "application/json"), withLabel);
  });

  $("file-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const file = $("file-input").files[0];
    if (!file) return showErrors($("file-errors"), ["Choose a CSV file first."]);
    if (file.size > schema.limits.max_upload_bytes) {
      return showErrors($("file-errors"), [`The file is larger than ${Math.round(schema.limits.max_upload_bytes / 1000)} KB.`]);
    }
    submit($("file-errors"), event.submitter, async () => post("/api/predict-file", await file.text(), "text/csv"));
  });

  $("new-prediction").addEventListener("click", () => {
    $("result-table").tBodies[0].replaceChildren();
    $("result-notes").replaceChildren();
    $("result-view").hidden = true;
    $("input-view").hidden = false;
    selectTab("form");
  });
})();
