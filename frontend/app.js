/* ============================================================
   Mapper Phase 3 - Frontend Application
   ============================================================ */

const API = "http://127.0.0.1:8000";

// ============================================================
// Global state
// ============================================================
let currentSessionId = null;
let currentMappings = [];
let reviewActions = [];
let allTargets = [];

// Memory management
let memoryRecords = [];
let pendingChanges = [];     // { action, data, _tempId }
let memoryStatusFilter = "ACTIVE";

// Saved tasks
let savedTasks = [];

// ============================================================
// Navigation
// ============================================================
document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", () => {
    const page = item.dataset.page;
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    item.classList.add("active");
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    document.getElementById(`page-${page}`).classList.add("active");

    if (page === "memory") loadMemoryRecords();
    if (page === "tasks") loadSavedTasks();
  });
});

// ============================================================
// Utility helpers
// ============================================================
function toast(el, msg, type = "info") {
  el.textContent = msg;
  el.className = `toast toast-${type}`;
  el.style.display = "block";
}

function hideToast(el) { el.style.display = "none"; }

function confClass(c) {
  if (c >= 0.8) return "conf-high";
  if (c >= 0.5) return "conf-med";
  return "conf-low";
}

function shortDate(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

async function loadAllTargets() {
  try {
    const res = await fetch(`${API}/targets/`);
    if (res.ok) {
      const data = await res.json();
      allTargets = data.targets;
    }
  } catch (e) { console.error("Failed to load targets:", e); }
}

function populateTargetSelect(selectEl, highlightCandidates) {
  const cands = new Set((highlightCandidates || []).map(c => c.target?.field_name || c.field_name));
  selectEl.innerHTML = '<option value="">Select target...</option>';
  allTargets.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t.field_name;
    const star = cands.has(t.field_name) ? "\u2B50 " : "";
    const cat = t.category ? ` [${t.category}]` : "";
    opt.textContent = `${star}${t.field_name}${cat}`;
    if (cands.has(t.field_name)) { opt.style.fontWeight = "bold"; opt.style.color = "#4f6ef7"; }
    selectEl.appendChild(opt);
  });
}

// ============================================================
// PAGE: Mapping
// ============================================================
const statusEl     = document.getElementById("status");
const fileInput    = document.getElementById("fileInput");
const tenantInput  = document.getElementById("tenantName");
const uploadBtn    = document.getElementById("uploadBtn");
const reviewSec    = document.getElementById("reviewSection");
const reviewBody   = document.getElementById("reviewBody");
const approveBtn   = document.getElementById("finalApproveBtn");
const cancelBtn    = document.getElementById("cancelReviewBtn");
const saveTaskBtn  = document.getElementById("saveTaskBtn");

uploadBtn.addEventListener("click", handleUpload);
approveBtn.addEventListener("click", finalizeReview);
cancelBtn.addEventListener("click", cancelReview);
saveTaskBtn.addEventListener("click", () => {
  document.getElementById("saveTaskModal").style.display = "flex";
  document.getElementById("saveTaskName").value = "";
  document.getElementById("saveTaskName").focus();
});

async function handleUpload() {
  const file = fileInput.files[0];
  const tenant = tenantInput.value.trim();
  if (!tenant) { toast(statusEl, "Tenant name is required", "error"); return; }
  if (!file)   { toast(statusEl, "Select a CSV file", "error"); return; }

  if (!allTargets.length) await loadAllTargets();

  try {
    toast(statusEl, "Profiling...", "info");
    const form = new FormData();
    form.append("file", file);

    const profRes = await fetch(`${API}/profile/`, { method: "POST", body: form });
    if (!profRes.ok) { toast(statusEl, "Profile failed", "error"); return; }
    const profData = await profRes.json();

    toast(statusEl, "Generating mappings...", "info");
    const mapRes = await fetch(`${API}/map/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profiles: profData.profiles, tenant_name: tenant }),
    });
    if (!mapRes.ok) {
      const err = await mapRes.json();
      toast(statusEl, err.detail || "Mapping failed", "error");
      return;
    }
    const mapData = await mapRes.json();

    // Start review session
    const sesRes = await fetch(`${API}/review/start/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mappings: mapData, tenant_name: tenant }),
    });
    if (!sesRes.ok) { toast(statusEl, "Session start failed", "error"); return; }
    const sesData = await sesRes.json();
    currentSessionId = sesData.session_id;

    toast(statusEl, `${mapData.mappings.length} mappings generated`, "success");
    renderReview(mapData.mappings);
  } catch (e) { toast(statusEl, e.message, "error"); }
}

function renderReview(mappings) {
  currentMappings = mappings;
  reviewActions = new Array(mappings.length);
  reviewBody.innerHTML = "";

  mappings.forEach((m, i) => {
    const target = m.selected_target ? m.selected_target.field_name : "None";
    const conf = m.confidence || 0;

    const tr = document.createElement("tr");
    tr.id = `rev-row-${i}`;
    tr.innerHTML = `
      <td>${m.source_column}</td>
      <td>${target}</td>
      <td><span class="conf-badge ${confClass(conf)}">${(conf * 100).toFixed(0)}%</span></td>
      <td class="actions-cell">
        <button class="btn btn-sm btn-success" onclick="approveRow(${i})" title="Approve"><i class="fas fa-check"></i></button>
        <button class="btn btn-sm btn-outline" onclick="rejectRow(${i})" title="Reject"><i class="fas fa-pen"></i></button>
      </td>`;
    reviewBody.appendChild(tr);
  });

  reviewSec.style.display = "block";
  reviewSec.scrollIntoView({ behavior: "smooth" });
}

window.approveRow = function(i) {
  const tr = document.getElementById(`rev-row-${i}`);
  tr.className = "row-approved";
  reviewActions[i] = {
    action: "approve",
    source_column: currentMappings[i].source_column,
    original_target: currentMappings[i].selected_target?.field_name || null,
  };
  // Remove correction row if exists
  const corr = document.getElementById(`corr-row-${i}`);
  if (corr) corr.remove();
};

window.rejectRow = function(i) {
  const tr = document.getElementById(`rev-row-${i}`);
  tr.className = "row-rejected";
  // Remove existing correction row
  let corr = document.getElementById(`corr-row-${i}`);
  if (corr) corr.remove();

  corr = document.createElement("tr");
  corr.id = `corr-row-${i}`;
  corr.className = "correction-row";
  corr.innerHTML = `
    <td colspan="4" style="padding-left:28px">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <select id="corr-sel-${i}" style="flex:1;min-width:200px"></select>
        <button class="btn btn-sm btn-primary" onclick="saveCorr(${i})">Save</button>
      </div>
    </td>`;
  tr.after(corr);

  populateTargetSelect(corr.querySelector(`#corr-sel-${i}`), currentMappings[i].candidates);
};

window.saveCorr = function(i) {
  const val = document.getElementById(`corr-sel-${i}`).value;
  if (!val) { alert("Select a target field"); return; }
  reviewActions[i] = {
    action: "reject",
    source_column: currentMappings[i].source_column,
    original_target: currentMappings[i].selected_target?.field_name || null,
    corrected_target: val,
    notes: "",
  };
  toast(statusEl, "Correction saved", "success");
};

async function finalizeReview() {
  if (!currentSessionId) { toast(statusEl, "No active session", "error"); return; }
  const tenant = tenantInput.value.trim();

  const reviews = currentMappings.map((m, i) => {
    if (reviewActions[i]) return reviewActions[i];
    return {
      source_column: m.source_column,
      original_target: m.selected_target?.field_name || null,
      action: "approve",
      corrected_target: null,
      notes: "",
    };
  });

  try {
    toast(statusEl, "Submitting...", "info");
    const subRes = await fetch(`${API}/review/submit/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: currentSessionId, tenant_name: tenant, reviews }),
    });
    if (!subRes.ok) {
      const err = await subRes.json();
      toast(statusEl, err.detail || "Submit failed", "error");
      return;
    }

    const finRes = await fetch(`${API}/review/finalize/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: currentSessionId }),
    });
    if (!finRes.ok) {
      const err = await finRes.json();
      toast(statusEl, err.detail || "Finalize failed", "error");
      return;
    }

    const finData = await finRes.json();
    downloadCSV(finData.mappings, tenant);
    toast(statusEl, `Done! ${finData.corrections_saved} corrections saved`, "success");
    resetReview();
  } catch (e) { toast(statusEl, e.message, "error"); }
}

function resetReview() {
  reviewSec.style.display = "none";
  currentSessionId = null;
  currentMappings = [];
  reviewActions = [];
}

function cancelReview() {
  if (confirm("Cancel review? Progress will be lost.")) resetReview();
}

function downloadCSV(mappings, tenant) {
  const hdr = ["source_column", "target_field", "status", "confidence"];
  const rows = mappings.map(m => [m.source_column, m.target_field || "", m.status || "approved", m.confidence || ""].join(","));
  const csv = [hdr.join(","), ...rows].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = `mappings_${tenant}_${new Date().toISOString().split("T")[0]}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}


// ============================================================
// PAGE: Memory Management
// ============================================================
const memBody       = document.getElementById("memoryBody");
const addMemBtn     = document.getElementById("addMemoryBtn");
const commitBtn     = document.getElementById("commitMemoryBtn");
const pendBanner    = document.getElementById("pendingBanner");
const pendText      = document.getElementById("pendingText");
const pendBadge     = document.getElementById("pendingBadge");
const clearPendBtn  = document.getElementById("clearPendingBtn");
const memModal      = document.getElementById("memoryModal");
const closeMemModal = document.getElementById("closeMemoryModal");
const cancelMemMod  = document.getElementById("cancelMemoryModal");
const saveMemMod    = document.getElementById("saveMemoryModal");

// Filter buttons
document.getElementById("memFilterActive").addEventListener("click", () => { memoryStatusFilter = "ACTIVE"; refreshMemoryFilter(); loadMemoryRecords(); });
document.getElementById("memFilterDisabled").addEventListener("click", () => { memoryStatusFilter = "DISABLED"; refreshMemoryFilter(); loadMemoryRecords(); });

function refreshMemoryFilter() {
  document.querySelectorAll(".toggle-btn[data-status]").forEach(b => {
    b.classList.toggle("active", b.dataset.status === memoryStatusFilter);
  });
}

async function loadMemoryRecords() {
  if (!allTargets.length) await loadAllTargets();
  try {
    const res = await fetch(`${API}/memory/list/?status=${memoryStatusFilter}`);
    if (!res.ok) return;
    const data = await res.json();
    memoryRecords = data.records;
    renderMemoryTable();
  } catch (e) { console.error("Load memory failed:", e); }
}

function renderMemoryTable() {
  memBody.innerHTML = "";

  // Render pending CREATEs at top
  pendingChanges.filter(c => c.action === "CREATE").forEach(c => {
    const tr = document.createElement("tr");
    tr.className = "pending-create";
    tr.innerHTML = `
      <td>${c.data.source_column}</td>
      <td>${c.data.target_field}</td>
      <td>${c.data.tenant_name || "-"}</td>
      <td>${c.data.category || "-"}</td>
      <td>manual</td>
      <td>-</td>
      <td>Pending</td>
      <td><span class="pill pill-active">NEW</span></td>
      <td class="actions-cell">
        <button class="btn btn-xs btn-ghost" onclick="undoPending('${c._tempId}')" title="Undo"><i class="fas fa-undo"></i></button>
      </td>`;
    memBody.appendChild(tr);
  });

  memoryRecords.forEach(rec => {
    const id = rec._id;
    const isPendingDelete = pendingChanges.some(c => c.action === "DELETE" && c.data.memory_id === id);
    const pendingUpdate = pendingChanges.find(c => c.action === "UPDATE" && c.data.memory_id === id);
    let cls = "";
    if (isPendingDelete) cls = "pending-delete";
    else if (pendingUpdate) cls = "pending-update";

    const status = rec.status || "ACTIVE";
    const pillCls = status === "ACTIVE" ? "pill-active" : "pill-disabled";

    const tr = document.createElement("tr");
    tr.className = cls;
    tr.innerHTML = `
      <td>${rec.source_column}</td>
      <td>${pendingUpdate ? pendingUpdate.data.new_target_field : rec.target_field}</td>
      <td>${rec.tenant_name || "-"}</td>
      <td>${rec.category || "-"}</td>
      <td>${rec.memory_source || "-"}</td>
      <td>${rec.usage_count || 1}</td>
      <td>${shortDate(rec.created_date)}</td>
      <td><span class="pill ${pillCls}">${status}</span></td>
      <td class="actions-cell">
        ${status === "ACTIVE" && !isPendingDelete ? `
          <button class="btn btn-xs btn-ghost" onclick="editMemory('${id}')" title="Edit"><i class="fas fa-pen"></i></button>
          <button class="btn btn-xs btn-ghost" onclick="deleteMemory('${id}')" title="Delete"><i class="fas fa-trash"></i></button>
        ` : ""}
        ${isPendingDelete || pendingUpdate ? `<button class="btn btn-xs btn-ghost" onclick="undoPendingById('${id}')" title="Undo"><i class="fas fa-undo"></i></button>` : ""}
      </td>`;
    memBody.appendChild(tr);
  });

  updatePendingUI();
}

function updatePendingUI() {
  const count = pendingChanges.length;
  pendBanner.style.display = count > 0 ? "flex" : "none";
  commitBtn.style.display = count > 0 ? "inline-flex" : "none";
  pendText.textContent = `${count} pending change${count !== 1 ? "s" : ""}`;
  pendBadge.textContent = count;
}

// Add new memory
addMemBtn.addEventListener("click", () => {
  document.getElementById("memoryModalTitle").textContent = "New Memory Record";
  document.getElementById("modalMode").value = "CREATE";
  document.getElementById("modalMemoryId").value = "";
  document.getElementById("modalSourceCol").value = "";
  document.getElementById("modalTenant").value = "";
  document.getElementById("modalCategory").value = "";
  populateTargetSelect(document.getElementById("modalTargetField"), []);
  document.getElementById("saveMemoryModal").textContent = "Add to Queue";
  memModal.style.display = "flex";
});

// Edit memory
window.editMemory = function(id) {
  const rec = memoryRecords.find(r => r._id === id);
  if (!rec) return;
  document.getElementById("memoryModalTitle").textContent = "Edit Memory Record";
  document.getElementById("modalMode").value = "UPDATE";
  document.getElementById("modalMemoryId").value = id;
  document.getElementById("modalSourceCol").value = rec.source_column;
  document.getElementById("modalTenant").value = rec.tenant_name || "";
  document.getElementById("modalCategory").value = rec.category || "";
  populateTargetSelect(document.getElementById("modalTargetField"), []);
  document.getElementById("modalTargetField").value = rec.target_field;
  document.getElementById("saveMemoryModal").textContent = "Queue Update";
  memModal.style.display = "flex";
};

// Delete memory
window.deleteMemory = function(id) {
  if (pendingChanges.some(c => c.action === "DELETE" && c.data.memory_id === id)) return;
  pendingChanges.push({ action: "DELETE", data: { memory_id: id }, _tempId: "del_" + id });
  renderMemoryTable();
};

// Undo pending
window.undoPending = function(tempId) {
  pendingChanges = pendingChanges.filter(c => c._tempId !== tempId);
  renderMemoryTable();
};

window.undoPendingById = function(id) {
  pendingChanges = pendingChanges.filter(c => !(c.data.memory_id === id));
  renderMemoryTable();
};

clearPendBtn.addEventListener("click", () => {
  pendingChanges = [];
  renderMemoryTable();
});

// Save modal
saveMemMod.addEventListener("click", () => {
  const mode = document.getElementById("modalMode").value;
  const src = document.getElementById("modalSourceCol").value.trim();
  const tgt = document.getElementById("modalTargetField").value;
  const ten = document.getElementById("modalTenant").value.trim();
  const cat = document.getElementById("modalCategory").value.trim();

  if (!src) { alert("Source column is required"); return; }
  if (!tgt) { alert("Target field is required"); return; }

  if (mode === "CREATE") {
    pendingChanges.push({
      action: "CREATE",
      data: { source_column: src, target_field: tgt, tenant_name: ten, category: cat, memory_source: "manual" },
      _tempId: "new_" + Date.now(),
    });
  } else {
    const memId = document.getElementById("modalMemoryId").value;
    // Remove any existing pending update for same id
    pendingChanges = pendingChanges.filter(c => !(c.action === "UPDATE" && c.data.memory_id === memId));
    pendingChanges.push({
      action: "UPDATE",
      data: { memory_id: memId, source_column: src, new_target_field: tgt, tenant_name: ten, category: cat },
      _tempId: "upd_" + memId,
    });
  }

  memModal.style.display = "none";
  renderMemoryTable();
});

closeMemModal.addEventListener("click", () => { memModal.style.display = "none"; });
cancelMemMod.addEventListener("click", () => { memModal.style.display = "none"; });

// Final Commit
commitBtn.addEventListener("click", async () => {
  if (!pendingChanges.length) return;
  if (!confirm(`Commit ${pendingChanges.length} changes to database?`)) return;

  try {
    const res = await fetch(`${API}/memory/commit/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ changes: pendingChanges.map(c => ({ action: c.action, data: c.data })) }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert("Commit failed: " + (err.detail || "Unknown error"));
      return;
    }

    const result = await res.json();
    pendingChanges = [];
    await loadMemoryRecords();
    alert(`Committed: ${result.created} created, ${result.updated} updated, ${result.deleted} deleted`);
  } catch (e) {
    alert("Commit error: " + e.message);
  }
});


// ============================================================
// PAGE: Saved Tasks
// ============================================================
const tasksBody = document.getElementById("tasksBody");
const noTasks   = document.getElementById("noTasks");

async function loadSavedTasks() {
  try {
    const res = await fetch(`${API}/tasks/list/`);
    if (!res.ok) return;
    const data = await res.json();
    savedTasks = data.tasks;
    renderTasksTable();
  } catch (e) { console.error("Load tasks failed:", e); }
}

function renderTasksTable() {
  tasksBody.innerHTML = "";
  if (!savedTasks.length) {
    noTasks.style.display = "block";
    return;
  }
  noTasks.style.display = "none";

  savedTasks.forEach(t => {
    const statusCls = t.review_status === "COMPLETED" ? "pill-completed" : "pill-saved";
    const mapCount = Array.isArray(t.mapping_data) ? t.mapping_data.length : 0;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.task_name}</td>
      <td>${t.tenant_name}</td>
      <td>${mapCount}</td>
      <td><span class="pill ${statusCls}">${t.review_status}</span></td>
      <td>${shortDate(t.updated_date)}</td>
      <td class="actions-cell">
        ${t.review_status === "SAVED" ? `
          <button class="btn btn-xs btn-primary" onclick="resumeTask('${t._id}')"><i class="fas fa-play"></i> Resume</button>
          <button class="btn btn-xs btn-success" onclick="completeTask('${t._id}')"><i class="fas fa-check"></i></button>
        ` : ""}
        <button class="btn btn-xs btn-ghost" onclick="deleteTask('${t._id}')" title="Delete"><i class="fas fa-trash"></i></button>
      </td>`;
    tasksBody.appendChild(tr);
  });
}

// Save Task from review
const saveTaskModal   = document.getElementById("saveTaskModal");
const closeSaveModal  = document.getElementById("closeSaveTaskModal");
const cancelSaveTask  = document.getElementById("cancelSaveTask");
const confirmSaveTask = document.getElementById("confirmSaveTask");

closeSaveModal.addEventListener("click", () => { saveTaskModal.style.display = "none"; });
cancelSaveTask.addEventListener("click", () => { saveTaskModal.style.display = "none"; });

confirmSaveTask.addEventListener("click", async () => {
  const name = document.getElementById("saveTaskName").value.trim();
  if (!name) { alert("Task name is required"); return; }

  const tenant = tenantInput.value.trim();
  // Build mapping_data from current state
  const mappingData = currentMappings.map((m, i) => {
    const action = reviewActions[i];
    return {
      source_column: m.source_column,
      selected_target: m.selected_target?.field_name || null,
      confidence: m.confidence,
      decision: m.decision,
      explanation: m.explanation,
      review_action: action ? action.action : null,
      corrected_target: action?.corrected_target || null,
    };
  });

  try {
    const res = await fetch(`${API}/tasks/save/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_name: name, tenant_name: tenant, mapping_data: mappingData }),
    });
    if (!res.ok) {
      const err = await res.json();
      alert("Save failed: " + (err.detail || "Unknown error"));
      return;
    }
    saveTaskModal.style.display = "none";
    toast(statusEl, "Task saved successfully", "success");
  } catch (e) { alert("Save error: " + e.message); }
});

// Resume task
window.resumeTask = async function(id) {
  if (!allTargets.length) await loadAllTargets();
  try {
    const res = await fetch(`${API}/tasks/${id}/`);
    if (!res.ok) { alert("Failed to load task"); return; }
    const task = await res.json();

    // Reconstruct mappings from mapping_data (full shape for API)
    const getTarget = (fieldName) => {
      if (!fieldName) return null;
      const t = allTargets.find(x => x.field_name === fieldName);
          return t ? { field_name: t.field_name, description: t.description || "", category: t.category || "", required: t.required } : { field_name: fieldName, description: "", category: "", required: "" };
    };
    const mappingsForApi = task.mapping_data.map(d => ({
      source_column: d.source_column,
      selected_target: d.selected_target ? getTarget(d.selected_target) : null,
      confidence: d.confidence || 0,
      decision: d.decision || "needs_review",
      explanation: d.explanation || "",
      candidates: [],
    }));

    // Create a real review session so Approve & Download works
    const startRes = await fetch(`${API}/review/start/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mappings: { mappings: mappingsForApi }, tenant_name: task.tenant_name }),
    });
    if (!startRes.ok) {
      alert("Failed to start review session");
      return;
    }
    const startData = await startRes.json();
    currentSessionId = startData.session_id;

    // Switch to mapping page
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    document.querySelector('[data-page="mapping"]').classList.add("active");
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    document.getElementById("page-mapping").classList.add("active");

    tenantInput.value = task.tenant_name;
    toast(statusEl, `Resumed: ${task.task_name}`, "info");

    // Reconstruct for display (selected_target as object with field_name)
    const fakeMappings = task.mapping_data.map(d => ({
      source_column: d.source_column,
      selected_target: d.selected_target ? { field_name: d.selected_target } : null,
      confidence: d.confidence || 0,
      decision: d.decision || "needs_review",
      explanation: d.explanation || "",
      candidates: [],
    }));
    renderReview(fakeMappings);

    // Restore review actions
    task.mapping_data.forEach((d, i) => {
      if (d.review_action === "approve") approveRow(i);
      else if (d.review_action === "reject" && d.corrected_target) {
        rejectRow(i);
        setTimeout(() => {
          const sel = document.getElementById(`corr-sel-${i}`);
          if (sel) sel.value = d.corrected_target;
          saveCorr(i);
        }, 50);
      }
    });
  } catch (e) { alert("Resume error: " + e.message); }
};

// Complete task
window.completeTask = async function(id) {
  try {
    const res = await fetch(`${API}/tasks/${id}/status/`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: id, review_status: "COMPLETED" }),
    });
    if (res.ok) await loadSavedTasks();
  } catch (e) { alert("Error: " + e.message); }
};

// Delete task
window.deleteTask = async function(id) {
  if (!confirm("Delete this task?")) return;
  try {
    const res = await fetch(`${API}/tasks/${id}/`, { method: "DELETE" });
    if (res.ok) await loadSavedTasks();
  } catch (e) { alert("Error: " + e.message); }
};


// ============================================================
// PAGE: Training
// ============================================================
const trainingForm    = document.getElementById("trainingForm");
const trainingResults = document.getElementById("trainingResults");

if (trainingForm) {
  trainingForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = document.getElementById("trainingFile").files[0];
    const tenant = document.getElementById("trainingTenant").value.trim();
    if (!tenant) { alert("Tenant name is required"); return; }
    if (!file) { alert("Select a file"); return; }

    try {
      trainingResults.innerHTML = '<div class="toast toast-info">Processing...</div>';
      const fd = new FormData();
      fd.append("file", file);
      fd.append("tenant_name", tenant);

      const res = await fetch(`${API}/train/ingest/`, { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json();
        trainingResults.innerHTML = `<div class="toast toast-error">${err.detail}</div>`;
        return;
      }

      const result = await res.json();
      const acc = (result.accuracy * 100).toFixed(1);
      trainingResults.innerHTML = `
        <div class="toast toast-success">Ingestion complete</div>
        <div class="training-grid">
          <div class="stat-card"><div class="stat-value">${result.total_rows}</div><div class="stat-label">Total Rows</div></div>
          <div class="stat-card"><div class="stat-value">${result.agent_correct}</div><div class="stat-label">Correct</div></div>
          <div class="stat-card"><div class="stat-value">${result.mismatches_saved}</div><div class="stat-label">Saved</div></div>
          <div class="stat-card"><div class="stat-value">${acc}%</div><div class="stat-label">Accuracy</div></div>
        </div>`;
    } catch (e) {
      trainingResults.innerHTML = `<div class="toast toast-error">${e.message}</div>`;
    }
  });
}

// ============================================================
// Init
// ============================================================
loadAllTargets();
