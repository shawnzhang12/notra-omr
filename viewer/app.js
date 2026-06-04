const state = {
  toolkit: null,
  notes: [],
  noteMap: new Map(),
  selectedNoteId: null,
  svgLinkedCount: 0,
  selectedFixture: null,
};

const dom = {
  fixtureSelect: document.getElementById("fixture-select"),
  xmlPath: document.getElementById("xml-path"),
  loadBtn: document.getElementById("load-btn"),
  filter: document.getElementById("filter"),
  status: document.getElementById("status"),
  stats: document.getElementById("stats"),
  notation: document.getElementById("notation"),
  sourceImage: document.getElementById("source-image"),
  sourceCaption: document.getElementById("source-caption"),
  sourceWrap: document.getElementById("source-wrap"),
  tableBody: document.querySelector("#note-table tbody"),
  selectionEmpty: document.getElementById("selection-empty"),
  selectionJson: document.getElementById("selection-json"),
};

const FIXTURES = [
  {
    group: "Demo",
    label: "Complex All Features",
    xml: "../artifacts/executive/step-7/examples/02_complex_all_features.musicxml",
  },
  ...[
    ["01 Fanfare", "01-fanfare"],
    ["02 Skating", "02-skating"],
    ["03 The Elephant's Waltz", "03-the_elephants_waltz"],
    ["04 The Tired Tortoise", "04-the_tired_tortoise"],
    ["05 The Whale's Song", "05-the_whales_song"],
    ["06 March", "06-march"],
    ["07 Sitting In The Shade", "07-sitting_in_the_shade"],
    ["Feuillard Exercise 1", "feuillard_exercise_1"],
    ["Left Hand Exercise 1", "left_hand_exercise_1"],
    ["Lista Trio Sonata In D Minor", "lista-trio_sonata_in_d_minor"],
    ["Suzuki Book 1 Piece 1", "suzuki_book_1_piece_1"],
    ["Suzuki Book 1 Piece 2", "suzuki_book_1_piece_2"],
    ["Suzuki Book 1 Piece 3", "suzuki_book_1_piece_3"],
    ["Suzuki Book 1 Piece 4", "suzuki_book_1_piece_4"],
    ["Suzuki Book 1 Piece 5", "suzuki_book_1_piece_5"],
    ["Suzuki Book 1 Piece 6", "suzuki_book_1_piece_6"],
    ["Suzuki Book 1 Piece 7", "suzuki_book_1_piece_7"],
    ["Untitled Score", "untitled_score"],
  ].map(([label, slug]) => ({
    group: "Cello Golden Fixtures",
    label,
    xml: `../tests/fixtures/golden/cello/${slug}.musicxml`,
    image: `../tests/fixtures/images/cello/${slug}/page-001.png`,
  })),
];

function setStatus(text, ok = true) {
  dom.status.textContent = text;
  dom.status.style.color = ok ? "#166534" : "#b91c1c";
}

function populateFixtureSelect() {
  dom.fixtureSelect.innerHTML = "";
  const manual = document.createElement("option");
  manual.value = "";
  manual.textContent = "Custom path";
  dom.fixtureSelect.appendChild(manual);

  const groups = new Map();
  for (const fixture of FIXTURES) {
    if (!groups.has(fixture.group)) {
      const group = document.createElement("optgroup");
      group.label = fixture.group;
      groups.set(fixture.group, group);
      dom.fixtureSelect.appendChild(group);
    }

    const option = document.createElement("option");
    option.value = fixture.xml;
    option.textContent = fixture.label;
    groups.get(fixture.group).appendChild(option);
  }

  const initialFixture = FIXTURES.find((fixture) => fixture.xml === dom.xmlPath.value);
  if (initialFixture) {
    state.selectedFixture = initialFixture;
    dom.fixtureSelect.value = initialFixture.xml;
    updateSourceImage(initialFixture);
  }
}

function updateSourceImage(fixture) {
  if (!fixture || !fixture.image) {
    dom.sourceImage.removeAttribute("src");
    dom.sourceWrap.classList.add("empty");
    dom.sourceCaption.textContent = "No source page image for this selection.";
    return;
  }

  dom.sourceWrap.classList.remove("empty");
  dom.sourceImage.src = fixture.image;
  dom.sourceCaption.textContent = fixture.image;
}

function selectFixtureByPath(xmlPath) {
  const fixture = FIXTURES.find((item) => item.xml === xmlPath) || null;
  state.selectedFixture = fixture;
  dom.fixtureSelect.value = fixture ? fixture.xml : "";
  updateSourceImage(fixture);
}

function parseMusicXmlNotes(xmlText) {
  const parser = new DOMParser();
  const xml = parser.parseFromString(xmlText, "application/xml");
  const notes = [];
  let index = 0;

  for (const part of Array.from(xml.querySelectorAll("part"))) {
    let currentDivisions = 1;
    for (const measure of Array.from(part.querySelectorAll(":scope > measure"))) {
      const measureNo = measure.getAttribute("number") || "?";
      const divisionsText = measure.querySelector(":scope > attributes > divisions")?.textContent;
      if (divisionsText && Number.isFinite(Number(divisionsText)) && Number(divisionsText) > 0) {
        currentDivisions = Number(divisionsText);
      }

      for (const note of Array.from(measure.querySelectorAll(":scope > note"))) {
        index += 1;
        const id = note.getAttribute("id") || `anon-note-${index}`;
        const isRest = !!note.querySelector("rest");
        const step = note.querySelector("pitch > step")?.textContent || "R";
        const alterRaw = note.querySelector("pitch > alter")?.textContent;
        const alter = alterRaw ? Number(alterRaw) : 0;
        const octave = note.querySelector("pitch > octave")?.textContent || "";
        const durationUnits = Number(note.querySelector("duration")?.textContent || "0");
        const type = note.querySelector("type")?.textContent || "?";
        const dotCount = note.querySelectorAll("dot").length;
        const fraction = unitsToFraction(durationUnits, currentDivisions);
        const duration = `${type}${dotCount ? ".".repeat(dotCount) : ""} (${fraction})`;
        const ties = Array.from(note.querySelectorAll("tie")).map((node) => node.getAttribute("type"));
        const slurs = Array.from(note.querySelectorAll("notations > slur")).map((node) =>
          node.getAttribute("type")
        );
        const beams = Array.from(note.querySelectorAll("beam")).map((node) => node.textContent?.trim() || "");
        const articulations = Array.from(note.querySelectorAll("notations > articulations > *")).map(
          (node) => node.tagName
        );
        const lyric = note.querySelector("lyric > text")?.textContent || null;
        const fingering = note.querySelector("notations > technical > fingering")?.textContent || null;
        const chord = !!note.querySelector("chord");
        const tuplet = note.querySelector("notations > tuplet")?.getAttribute("type") || null;

        const pitch = isRest
          ? "rest"
          : `${step}${alter === 0 ? "" : alter > 0 ? `+${alter}` : `${alter}`}${octave}`;

        const flags = [
          chord ? "chord" : null,
          ties.length ? `ties:${ties.join("/")}` : null,
          slurs.length ? `slurs:${slurs.join("/")}` : null,
          beams.length ? `beams:${beams.join("/")}` : null,
          tuplet ? `tuplet:${tuplet}` : null,
          articulations.length ? `art:${articulations.join(",")}` : null,
          lyric ? "lyric" : null,
          fingering ? `fing:${fingering}` : null,
        ]
          .filter(Boolean)
          .join(" | ");

        notes.push({
          id,
          pitch,
          duration,
          durationUnits,
          divisions: currentDivisions,
          measure: measureNo,
          lyric,
          fingering,
          ties,
          slurs,
          beams,
          tuplet,
          articulations,
          chord,
          flags,
          index,
        });
      }
    }
  }

  return notes;
}

function unitsToFraction(units, divisions) {
  if (!Number.isFinite(units) || !Number.isFinite(divisions) || divisions <= 0) {
    return "?";
  }
  const numerator = units;
  const denominator = divisions * 4;
  const reduced = reduceFraction(numerator, denominator);
  return `${reduced[0]}/${reduced[1]}`;
}

function reduceFraction(numerator, denominator) {
  let a = Math.abs(numerator);
  let b = Math.abs(denominator);
  while (b !== 0) {
    const t = b;
    b = a % b;
    a = t;
  }
  const gcd = a || 1;
  return [numerator / gcd, denominator / gcd];
}

function escapeCssId(value) {
  if (window.CSS && typeof window.CSS.escape === "function") {
    return window.CSS.escape(value);
  }
  return value.replace(/([#.;?+*~':"!^$\[\]()=>|/@])/g, "\\$1");
}

function renderNoteTable(notes) {
  dom.tableBody.innerHTML = "";

  for (const note of notes) {
    const row = document.createElement("tr");
    row.dataset.noteId = note.id;

    row.innerHTML = `
      <td>${note.measure}</td>
      <td>${note.id}</td>
      <td>${note.pitch}</td>
      <td>${note.duration}</td>
      <td>${note.flags || "-"}</td>
    `;

    row.addEventListener("click", () => selectNote(note.id));
    dom.tableBody.appendChild(row);
  }

  const unlinked = Math.max(0, state.notes.length - state.svgLinkedCount);
  dom.stats.textContent = `${notes.length}/${state.notes.length} shown | ${state.svgLinkedCount} linked | ${unlinked} unlinked`;
}

function clearSelection() {
  state.selectedNoteId = null;
  dom.selectionEmpty.hidden = false;
  dom.selectionJson.hidden = true;

  for (const row of dom.tableBody.querySelectorAll("tr")) {
    row.classList.remove("active");
  }

  for (const glyph of dom.notation.querySelectorAll(".glyph-active")) {
    glyph.classList.remove("glyph-active");
  }
}

function selectNote(noteId) {
  const note = state.noteMap.get(noteId);
  if (!note) return;

  state.selectedNoteId = noteId;
  dom.selectionEmpty.hidden = true;
  dom.selectionJson.hidden = false;
  dom.selectionJson.textContent = JSON.stringify(note, null, 2);

  for (const row of dom.tableBody.querySelectorAll("tr")) {
    row.classList.toggle("active", row.dataset.noteId === noteId);
  }

  for (const glyph of dom.notation.querySelectorAll(".glyph-active")) {
    glyph.classList.remove("glyph-active");
  }

  const glyphs = dom.notation.querySelectorAll(`[data-linked-note-id="${escapeCssId(noteId)}"]`);
  glyphs.forEach((node) => node.classList.add("glyph-active"));
}

function applyFilter() {
  const raw = dom.filter.value.trim().toLowerCase();
  if (!raw) {
    renderNoteTable(state.notes);
    if (state.selectedNoteId) selectNote(state.selectedNoteId);
    return;
  }

  const filtered = state.notes.filter((note) => {
    const haystack = `${note.id} ${note.pitch} ${note.flags} ${note.lyric || ""}`.toLowerCase();
    return haystack.includes(raw);
  });
  renderNoteTable(filtered);
}

function bindGlyphLinks() {
  const noteIds = state.notes.map((note) => note.id);
  let linked = 0;

  for (const noteId of noteIds) {
    const direct = dom.notation.querySelector(`#${escapeCssId(noteId)}`);
    if (direct) {
      direct.dataset.linkedNoteId = noteId;
      linked += 1;
    }
  }

  if (linked >= Math.max(1, Math.floor(noteIds.length * 0.5))) {
    state.svgLinkedCount = linked;
    return;
  }

  // Fallback: sequentially map note-like SVG groups to note ids.
  const candidates = Array.from(
    dom.notation.querySelectorAll("g.note[id], g[class*='note'][id], [class*='notehead'][id]")
  );
  const uniqueCandidates = candidates.filter((el, idx) => candidates.indexOf(el) === idx);
  const maxLinks = Math.min(uniqueCandidates.length, noteIds.length);

  for (let index = 0; index < maxLinks; index += 1) {
    uniqueCandidates[index].dataset.linkedNoteId = noteIds[index];
  }

  state.svgLinkedCount = maxLinks;
}

function hookNotationClicks() {
  dom.notation.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;

    const linked = target.closest("[data-linked-note-id]");
    if (linked && linked instanceof HTMLElement) {
      selectNote(linked.dataset.linkedNoteId);
      return;
    }

    const byId = target.closest("[id]");
    if (byId && byId.id && state.noteMap.has(byId.id)) {
      selectNote(byId.id);
    }
  });
}

async function renderWithVerovio(xmlText) {
  if (!window.verovio || !window.verovio.toolkit) {
    throw new Error("Verovio toolkit not available yet");
  }

  if (!state.toolkit) {
    state.toolkit = new window.verovio.toolkit();
  }

  state.toolkit.setOptions({
    scale: 42,
    pageWidth: 2200,
    adjustPageHeight: true,
    breaks: "none",
  });

  const loaded = state.toolkit.loadData(xmlText);
  if (loaded === false) {
    throw new Error("Verovio failed to load MusicXML data");
  }

  const svg = state.toolkit.renderToSVG(1, true);
  dom.notation.innerHTML = svg;
}

async function loadScore() {
  const xmlPath = dom.xmlPath.value.trim();
  if (!xmlPath) return;

  clearSelection();
  selectFixtureByPath(xmlPath);
  setStatus("Loading MusicXML...", true);

  try {
    const response = await fetch(xmlPath);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const xmlText = await response.text();

    state.notes = parseMusicXmlNotes(xmlText);
    state.noteMap = new Map(state.notes.map((note) => [note.id, note]));

    await renderWithVerovio(xmlText);
    bindGlyphLinks();
    renderNoteTable(state.notes);

    setStatus(
      `Loaded ${state.notes.length} notes. Linked ${state.svgLinkedCount} glyphs to note IDs.`,
      true
    );
  } catch (error) {
    console.error(error);
    dom.notation.innerHTML = "";
    dom.tableBody.innerHTML = "";
    dom.stats.textContent = "";
    setStatus(`Load failed: ${error.message}`, false);
  }
}

function bootstrap() {
  populateFixtureSelect();
  hookNotationClicks();
  dom.fixtureSelect.addEventListener("change", () => {
    const xmlPath = dom.fixtureSelect.value;
    if (!xmlPath) {
      state.selectedFixture = null;
      updateSourceImage(null);
      return;
    }
    dom.xmlPath.value = xmlPath;
    selectFixtureByPath(xmlPath);
    loadScore();
  });
  dom.loadBtn.addEventListener("click", loadScore);
  dom.xmlPath.addEventListener("input", () => selectFixtureByPath(dom.xmlPath.value.trim()));
  dom.filter.addEventListener("input", applyFilter);

  // Defer initial load slightly so the Verovio script can finish initializing.
  setTimeout(() => {
    loadScore();
  }, 100);
}

window.addEventListener("DOMContentLoaded", bootstrap);
