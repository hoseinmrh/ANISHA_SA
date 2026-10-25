const root = document.querySelector('#app');
const trialId = location.pathname.split('/')[2];
const agentNames = { coding_agent: 'Coding', writing_agent: 'Writing' };
const agents = Object.keys(agentNames);
let token = '';
let state = null;
let selected = '';
let detailPanel = '';
let error = '';
let busy = false;
let loggingFailed = false;
let recorder = null;
let recordingStream = null;
let recordingContext = null;
let activation = null;
let recording = false;
let recordingPurpose = 'briefing';
let recordingAgent = '';
let speech = { coding_agent: null, writing_agent: null };
let proposals = { coding_agent: null, writing_agent: null };
let followupSpeech = { coding_agent: null, writing_agent: null };
let followupResult = { coding_agent: null, writing_agent: null };
let followupDraft = { coding_agent: '', writing_agent: '' };
const openedEvents = new Set();
const openedRecords = new Set();
const seenCues = new Set(trialId ? JSON.parse(sessionStorage.getItem(`separate-cues:${trialId}`) || '[]') : []);
let cueNotice = '';

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
const micIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10a7 7 0 0 0 14 0M12 17v5m-4 0h8"/></svg>';
const stopIcon = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
const shortTime = (value) => new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const agentForEvent = (event) => event.task_id?.endsWith(':coding') ? 'coding_agent' : event.task_id?.endsWith(':writing') ? 'writing_agent' : null;
const isFinalFile = (record) => ['code_artifact', 'markdown_artifact'].includes(record.kind);

async function api(path, method = 'GET', body, researcher = false) {
  const headers = { [researcher ? 'X-ANISHA-Researcher-Token' : 'X-ANISHA-Study-Token']: researcher ? researcherToken : token };
  if (body !== undefined) headers['Content-Type'] = body instanceof Blob ? 'application/octet-stream' : 'application/json';
  const response = await fetch(path, { method, headers, body: body === undefined ? undefined : body instanceof Blob ? body : JSON.stringify(body), cache: 'no-store' });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((item) => `${item.loc?.slice(1).join('.') || 'Input'}: ${item.msg}`).join('; ')
      : typeof payload.detail === 'object' && payload.detail !== null
        ? JSON.stringify(payload.detail)
        : payload.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json();
}
const base = `/study/trials/${trialId}`;

async function refresh() {
  if (!trialId || !token || recording || busy) return;
  try {
    const next = await api(`${base}/state`);
    const changed = JSON.stringify(next) !== JSON.stringify(state);
    state = next;
    if (changed) {
      await announceCues(next);
      render();
    }
  } catch (cause) {
    error = cause.message;
    render();
  }
}

async function interact(eventType, targetId, layer) {
  try {
    return await api(`${base}/interactions`, 'POST', {
      event_type: eventType, target_id: targetId, ui_layer: layer, client_ts_ms: Date.now(),
    });
  } catch (cause) {
    loggingFailed = true;
    error = `The interaction log could not be saved: ${cause.message}. Tell the researcher.`;
    render();
    throw cause;
  }
}

async function announceCues(next) {
  for (const event of next.events) {
    if (!event.cue_id || seenCues.has(event.cue_id)) continue;
    seenCues.add(event.cue_id);
    sessionStorage.setItem(`separate-cues:${trialId}`, JSON.stringify([...seenCues]));
    cueNotice = 'New agent activity is available in the tabs.';
    try {
      if (recordingContext && recordingContext.state !== 'closed') {
        await recordingContext.resume();
        const oscillator = recordingContext.createOscillator();
        const gain = recordingContext.createGain();
        oscillator.frequency.value = 720;
        gain.gain.value = 0.07;
        oscillator.connect(gain).connect(recordingContext.destination);
        oscillator.start();
        oscillator.stop(recordingContext.currentTime + 0.17);
      }
      navigator.vibrate?.(160);
      await interact('cue_delivered', event.cue_id, 'agent_tabs');
    } catch { /* interact has already shown the logging failure */ }
  }
}

function selectAgent(agent) {
  if (recording || busy) return;
  selected = agent;
  detailPanel = '';
  sessionStorage.setItem(`separate-agent:${trialId}`, agent);
  void interact('agent_switch', agent, 'agent_tabs');
  render();
}

function sourceLabel(record) {
  if (record.record_id.endsWith('-S1')) return 'Latest supervisor instruction';
  if (record.record_id.endsWith('-S2')) return 'Earlier working note';
  if (record.record_id.endsWith('-S3')) return 'Verified analysis summary';
  return record.title;
}

function recordOwner(record) {
  if (record.kind === 'code_artifact') return 'coding_agent';
  if (['markdown_artifact', 'report_excerpt', 'draft_preview'].includes(record.kind)) return 'writing_agent';
  return null;
}

function renderContext() {
  const records = state.records.filter((record) => recordOwner(record) === null);
  return `<section class="context-section"><p class="label">Original request</p><p class="context-brief">${esc(state.participant_brief)}</p></section>
    <section class="context-section"><h3>Reference notes and instructions</h3><p class="small muted">Open a note to read its full text, date, and authority.</p>${records.map((record) => {
      const isOpen = openedRecords.has(record.record_id);
      return `<div class="source"><button type="button" data-action="source" data-id="${esc(record.record_id)}" aria-expanded="${isOpen}"><span>${esc(sourceLabel(record))}</span><small>${esc(record.title)}</small><span class="source-toggle" aria-hidden="true">${isOpen ? '−' : '+'}</span></button>${isOpen ? `<p>${esc(record.text)}</p>` : ''}</div>`;
    }).join('')}</section>`;
}

function renderOutputs(agent) {
  const records = state.records.filter((record) => recordOwner(record) === agent);
  if (!records.length) return '';
  return `<section class="output-thread" id="outputs" aria-label="${agentNames[agent]} documents">${records.map((record) => {
    const isOpen = openedRecords.has(record.record_id);
    const filename = record.kind === 'code_artifact' ? 'analysis.py' : record.kind === 'markdown_artifact' ? 'report.md' : '';
    const preview = record.text.length > 180 ? `${record.text.slice(0, 180)}…` : record.text;
    return `<div class="message message-agent"><span class="avatar" aria-hidden="true">${agentNames[agent][0]}</span><div class="bubble output-bubble"><div class="message-meta"><strong>${agentNames[agent]}</strong><span>${isFinalFile(record) ? 'Created file' : 'Working document'}</span></div><h3>${esc(record.title)}</h3><p class="output-preview">${esc(preview)}</p><button type="button" class="button secondary" data-action="source" data-id="${esc(record.record_id)}" aria-expanded="${isOpen}">${isOpen ? 'Hide document' : 'Open document'}</button>${isOpen ? `<div class="output-full"><pre>${esc(record.text)}</pre>${filename ? `<a download="${filename}" href="data:${record.kind === 'code_artifact' ? 'text/x-python' : 'text/markdown'};charset=utf-8,${encodeURIComponent(record.text)}">Download ${filename}</a>` : ''}</div>` : ''}</div></div>`;
  }).join('')}</section>`;
}

function renderDetailsPanel(agent) {
  if (!detailPanel) return '';
  return `<div class="details-overlay"><button type="button" class="details-scrim" data-action="close-details" aria-label="Close details"></button><aside class="details-sheet" role="dialog" aria-modal="true" aria-label="Project details"><header class="details-header"><div><p class="label">${agentNames[agent]} workspace</p><h2>Details</h2></div><button type="button" class="details-close" data-action="close-details" aria-label="Close details">×</button></header><nav class="details-tabs" aria-label="Detail sections"><button type="button" data-action="details" data-panel="context" aria-current="${detailPanel === 'context'}">Context</button><button type="button" data-action="details" data-panel="activity" aria-current="${detailPanel === 'activity'}">Activity</button></nav><div class="details-body">${detailPanel === 'context' ? renderContext() : renderFeed(agent)}</div></aside></div>`;
}

function renderFeed(agent) {
  const events = state.events.filter((event) => agentForEvent(event) === agent);
  const latest = events.at(-1);
  const latestLine = latest ? (latest.cue_id && !openedEvents.has(latest.event_id) ? 'New update available in activity.' : latest.text) : 'No activity yet.';
  return `<section class="conversation" aria-label="${agentNames[agent]} activity"><div class="thread-heading"><span class="online-dot"></span><span>Activity with ${agentNames[agent]}</span><span class="thread-count">${events.length} updates</span></div><div class="status-note"><span class="label">Latest status</span><p>${esc(latestLine)}</p></div><ol class="feed">${events.length ? events.map((event) => `<li class="message message-agent"><span class="avatar" aria-hidden="true">${agentNames[agent][0]}</span><div class="bubble"><div class="message-meta"><strong>${agentNames[agent]}</strong><time datetime="${new Date(event.occurred_at_ms).toISOString()}">${shortTime(event.occurred_at_ms)}</time></div>${event.cue_id && !openedEvents.has(event.event_id) ? `<p>New update ready to open.</p><button class="button secondary" data-action="event" data-id="${esc(event.event_id)}">Open update</button>` : `<p>${esc(event.text)}</p><button type="button" class="text-link" data-action="details" data-panel="context">Open reference notes</button>`}<span class="event-id">${esc(event.event_id)}</span></div></li>`).join('') : `<li class="empty-thread">${agentNames[agent]} has not posted an update yet.</li>`}</ol></section>`;
}

function renderBriefing(agent) {
  const confirmed = state.confirmed_agents.includes(agent);
  if (confirmed) return `<div class="system-message">${agentNames[agent]} briefing confirmed.${state.phase === 'awaiting_briefing' ? ' Brief the other agent to start both tasks.' : ''}</div>`;
  const current = speech[agent];
  const proposal = proposals[agent];
  return `<section class="briefing-thread"><div class="message message-agent"><span class="avatar" aria-hidden="true">${agentNames[agent][0]}</span><div class="bubble"><div class="message-meta"><strong>${agentNames[agent]}</strong></div><p>Tell me what you want me to work on. Record your briefing, review the words, then confirm the goal.</p></div></div>
    ${current ? `<div class="message message-user"><div class="bubble"><div class="message-meta"><strong>Your briefing</strong><span>Attempt ${current.activation_index}</span></div><label>Recognised words<textarea id="transcript" ${proposal ? 'disabled' : ''}>${esc(proposal?.reviewed_transcript ?? current.recognized_transcript)}</textarea></label>${!proposal ? `<button class="button secondary" data-action="review" ${current.status !== 'recognized' || busy || loggingFailed ? 'disabled' : ''}>Review proposal</button>` : ''}</div></div>` : ''}
    ${proposal ? `<div class="message message-agent"><span class="avatar" aria-hidden="true">${agentNames[agent][0]}</span><div class="bubble proposal-bubble"><div class="message-meta"><strong>Proposed ${agentNames[agent]} goal</strong></div><label>Goal<textarea id="goal">${esc(proposal.goal)}</textarea></label><p class="small">Sources: ${esc(proposal.source_refs.join(', '))}</p><p class="small">${esc(proposal.constraints.join(' '))}</p>${proposal.dependency ? `<p class="small">${esc(proposal.dependency)}</p>` : ''}<p class="small">${esc(proposal.approval_boundary)}</p><button class="button" data-action="confirm" ${busy || loggingFailed ? 'disabled' : ''}>Confirm ${agentNames[agent]} briefing</button></div></div>` : ''}
  </section>`;
}

function renderFollowup(agent) {
  if (['awaiting_briefing', 'probe', 'complete', 'technical_failure'].includes(state.phase)) return '';
  const heard = followupSpeech[agent];
  const answer = followupResult[agent];
  const showReview = heard?.status === 'recognized';
  if (!heard) return '';
  return `<section class="followup-thread" aria-label="Latest voice exchange">${heard.status !== 'recognized' ? '<div class="system-message error">No words were recognized. Record again.</div>' : `<div class="message message-user"><div class="bubble"><div class="message-meta"><strong>You</strong><span>Voice question</span></div><p>${esc(followupDraft[agent])}</p>${showReview ? `<details ${answer?.status === 'needs_clarification' ? 'open' : ''}><summary>Review or edit recognized words</summary><label>What ${agentNames[agent]} heard<textarea id="followup-transcript">${esc(followupDraft[agent])}</textarea></label><button type="button" class="button secondary" data-action="review-followup" ${busy || loggingFailed ? 'disabled' : ''}>Ask with reviewed words</button></details>` : ''}</div></div>`}
    ${answer ? `<div class="message message-agent"><span class="avatar" aria-hidden="true">${agentNames[agent][0]}</span><div class="bubble"><div class="message-meta"><strong>${agentNames[agent]}</strong></div><p role="status">${esc(answer.message)}</p>${answer.record_ids?.length ? `<div class="row">${answer.record_ids.map((id) => `<button type="button" class="button secondary small" data-action="source" data-id="${esc(id)}">Open ${esc(id)}</button>`).join('')}</div>` : ''}</div></div>` : ''}
  </section>`;
}

function renderComposer(agent) {
  if (state.phase === 'complete') return '';
  const briefing = state.phase === 'awaiting_briefing';
  if (briefing && state.confirmed_agents.includes(agent)) {
    return `<div class="chat-composer composer-idle"><p>${agentNames[agent]} is briefed. Select the other agent to continue.</p></div>`;
  }
  const prompt = briefing ? `Brief ${agentNames[agent]} with your voice` : `Ask ${agentNames[agent]} with your voice`;
  const buttonText = recording ? 'Stop and send' : briefing ? 'Record briefing' : 'Record question';
  return `<div class="chat-composer"><div class="composer-row"><span class="composer-prompt">${prompt}</span><button class="button microphone ${recording ? 'recording' : ''}" type="button" data-action="microphone" aria-label="${buttonText}" ${busy || loggingFailed ? 'disabled' : ''}>${recording ? stopIcon : micIcon}<span>${buttonText}</span></button></div><p class="composer-hint">${briefing ? 'Review the transcript and goal before confirming.' : 'Questions stay with this agent. Spoken choices still need an on-screen save.'}</p></div>`;
}

function renderDecision(agent) {
  const gateAgent = state.phase === 'blocker' ? 'coding_agent' : ['approval', 'final_review'].includes(state.phase) ? 'writing_agent' : null;
  const correctionAgent = state.correction_for_event_id ? (state.phase === 'awaiting_approval' ? 'coding_agent' : 'writing_agent') : null;
  let output = '';
  const suggested = followupResult[agent]?.suggested_choice_id;
  if (gateAgent === agent) {
    const title = state.phase === 'blocker' ? 'Choose a rule' : state.phase === 'approval' ? 'Review approval request' : 'Final report review';
    output += `<section class="card"><h2>${title}</h2><p>Inspect the activity and reference notes before deciding.</p><div class="row"><button type="button" class="text-link" data-action="details" data-panel="activity">Review activity</button><button type="button" class="text-link" data-action="details" data-panel="context">Open reference notes</button></div><div class="stack">${state.action_options.map((option) => `<label class="choice"><input name="decision" type="radio" value="${esc(option.choice_id)}" ${option.choice_id === suggested ? 'checked' : ''}>${esc(option.label)}</label>`).join('')}<label>Reason<textarea id="reason"></textarea></label>${state.phase === 'final_review' ? '<label>Exact claim needing correction<textarea id="claim"></textarea></label>' : ''}<label>Confidence (0–100)<input id="confidence" type="number" min="0" max="100"></label><button class="button" data-action="respond" ${busy || loggingFailed ? 'disabled' : ''}>Save first answer</button></div></section>`;
  }
  if (correctionAgent === agent) output += `<section class="card"><h2>Feedback and one correction</h2><p>First answer: ${esc(state.correction_previous_choice_label)}. ${esc(state.correction_previous_reason)}</p><p>See the latest feedback in the activity feed.</p><div class="stack">${state.correction_options.map((option) => `<label class="choice"><input name="correction" type="radio" value="${esc(option.choice_id)}" ${option.choice_id === suggested ? 'checked' : ''}>${esc(option.label)}</label>`).join('')}<label>Corrected reason<textarea id="correction-reason"></textarea></label><button class="button secondary" data-action="correct" ${busy || loggingFailed ? 'disabled' : ''}>Save correction</button></div></section>`;
  return output;
}

function render() {
  if (!trialId) return renderResearcher();
  if (state?.masked || state?.phase === 'probe') { detailPanel = ''; root.innerHTML = '<div class="mask" aria-label="Screen hidden for state probe"></div>'; return; }
  if (!state) { root.innerHTML = `<main><h1>Study interface</h1>${error ? `<p class="error">${esc(error)}</p>` : '<p>Loading trial…</p>'}</main>`; return; }
  if (state.phase === 'technical_failure') { root.innerHTML = '<main><h1>Period stopped</h1><p>Return the phone to the researcher.</p></main>'; return; }
  const current = selected && agents.includes(selected) ? selected : '';
  const task = state.tasks.find((item) => item.agent_id === current);
  const agentEvents = Object.fromEntries(agents.map((agent) => [agent, state.events.filter((event) => agentForEvent(event) === agent)]));
  const badge = (agent) => agentEvents[agent].filter((event) => event.cue_id && !openedEvents.has(event.event_id)).length + (state.phase === 'blocker' && agent === 'coding_agent' || ['approval', 'final_review'].includes(state.phase) && agent === 'writing_agent' ? 1 : 0);
  const threadContent = current && state.phase !== 'awaiting_briefing'
    ? renderOutputs(current) + renderFollowup(current) + renderDecision(current)
    : '';
  root.innerHTML = `<main class="participant-shell"><header class="top"><span class="brand-mark" aria-hidden="true">${current ? agentNames[current][0] : 'S'}</span><div class="brand-copy"><p class="label">${esc(state.interface_label)} · Separate Agents</p><h1 class="title">${current ? `${agentNames[current]} agent` : 'Choose an agent'}</h1></div><span class="status phase-status">${esc(state.phase.replaceAll('_', ' '))}</span></header>
    ${error ? `<p class="error" role="alert">${esc(error)}</p>` : ''}${cueNotice ? `<p class="notice">${esc(cueNotice)}</p>` : ''}
    ${!current ? `<section class="welcome-panel"><p class="label">Your conversations</p><h2>Coding and Writing</h2><p>Open an agent to brief it or review its activity.</p><div class="chooser">${agents.map((agent) => `<button class="agent-choice" data-action="select" data-agent="${agent}"><span class="avatar" aria-hidden="true">${agentNames[agent][0]}</span><span><strong>${agentNames[agent]}</strong><small>Open conversation</small></span><span class="chevron" aria-hidden="true">›</span></button>`).join('')}</div></section>` : `<nav class="tabs" aria-label="Agents">${agents.map((agent) => `<button class="tab" role="tab" aria-selected="${agent === current}" data-action="select" data-agent="${agent}" ${recording || busy ? 'disabled' : ''}><span class="tab-dot"></span>${agentNames[agent]} ${badge(agent) ? `<span class="badge">${badge(agent)}</span>` : ''}</button>`).join('')}</nav><div class="workspace-tools"><button type="button" data-action="details" data-panel="context">Context</button><button type="button" data-action="details" data-panel="activity">Activity ${badge(current) ? `<span class="badge">${badge(current)}</span>` : ''}</button></div><section class="task-banner"><div><p class="label">Current task</p><h2>${task ? esc(task.goal) : `${agentNames[current]} task`}</h2>${current === 'writing_agent' && task?.depends_on ? '<p class="small muted">Results section waits for verified Coding output.</p>' : ''}</div><span class="status ${esc(task?.state || '')}">${esc(task?.state?.replaceAll('_', ' ') || (state.confirmed_agents.includes(current) ? 'briefed' : 'awaiting briefing'))}</span></section>
    <div class="chat-content">${state.phase === 'awaiting_briefing' ? `<div class="system-message">Project brief and reference notes are in Context.</div>${renderBriefing(current)}` : threadContent || '<div class="system-message">No messages yet. Agent updates are in Activity.</div>'}${state.phase === 'complete' ? '<div class="system-message">Trial complete. Please return the phone to the researcher when finished.</div>' : ''}</div>
    <p class="footer">Study interface · ${esc(state.trial_id)}</p>${renderComposer(current)}${renderDetailsPanel(current)}`}</main>`;
  root.querySelectorAll('[data-action]').forEach((button) => button.addEventListener('click', action));
  root.querySelector('#followup-transcript')?.addEventListener('input', (event) => {
    followupDraft[current] = event.target.value;
  });
}

document.addEventListener?.('keydown', (event) => {
  if (!detailPanel) return;
  if (event.key === 'Escape') {
    const previous = detailPanel;
    detailPanel = '';
    render();
    root.querySelector(`.workspace-tools [data-panel="${previous}"]`)?.focus();
  } else if (event.key === 'Tab') {
    const focusable = [...root.querySelectorAll('.details-sheet button:not([disabled]), .details-sheet a[href]')];
    if (!focusable.length) return;
    if (event.shiftKey && document.activeElement === focusable[0]) {
      event.preventDefault();
      focusable.at(-1).focus();
    } else if (!event.shiftKey && document.activeElement === focusable.at(-1)) {
      event.preventDefault();
      focusable[0].focus();
    }
  }
});

async function action(event) {
  const button = event.currentTarget;
  const name = button.dataset.action;
  if (name === 'select') return selectAgent(button.dataset.agent);
  if (name === 'microphone') return recording ? stopRecording() : startRecording();
  if (name === 'close-details') {
    const previous = detailPanel;
    detailPanel = '';
    render();
    root.querySelector(`.workspace-tools [data-panel="${previous}"]`)?.focus();
    return;
  }
  if (name === 'details') {
    if (recording || busy || loggingFailed || !['context', 'activity'].includes(button.dataset.panel)) return;
    try {
      if (detailPanel !== button.dataset.panel) {
        await interact('ui_layer_open', button.dataset.panel, `${selected}_details`);
        detailPanel = button.dataset.panel;
      }
      render();
      root.querySelector('.details-close')?.focus();
    } catch { /* interact already reported the logging failure */ }
    return;
  }
  if (name === 'source') {
    const id = button.dataset.id;
    const record = state.records.find((item) => item.record_id === id);
    if (!record || loggingFailed) return;
    const isContext = recordOwner(record) === null;
    const isReference = !!button.closest('.followup-thread');
    try {
      if (isContext && detailPanel !== 'context') {
        await interact('ui_layer_open', 'context', `${selected}_details`);
        detailPanel = 'context';
      }
      if (openedRecords.has(id) && !isReference) openedRecords.delete(id);
      else if (!openedRecords.has(id)) {
        await interact('source_open', id, isContext ? `${selected}_context` : `${selected}_document`);
        openedRecords.add(id);
      }
      render();
      if (!isContext) root.querySelector('#outputs')?.scrollIntoView({ block: 'nearest' });
    } catch { /* interact already reported the logging failure */ }
    return;
  }
  if (name === 'event') {
    await interact('event_open', button.dataset.id, `${selected}_feed`);
    openedEvents.add(button.dataset.id); return render();
  }
  if (busy || loggingFailed) return;
  busy = true; error = ''; button.disabled = true;
  try {
    if (name === 'review') {
      const words = root.querySelector('#transcript').value.trim();
      if (!words) throw new Error('Review the recognized words first.');
      proposals[selected] = await api(`${base}/agents/${selected}/proposals`, 'POST', { transcription_id: speech[selected].transcription_id, reviewed_transcript: words });
    } else if (name === 'review-followup') {
      const words = root.querySelector('#followup-transcript').value.trim();
      if (!words) throw new Error('Review the recognized words first.');
      followupDraft[selected] = words;
      followupResult[selected] = await api(`${base}/agents/${selected}/followups`, 'POST', {
        transcription_id: followupSpeech[selected].transcription_id,
        reviewed_transcript: words,
      });
      if (followupResult[selected].revealed_event_id) openedEvents.add(followupResult[selected].revealed_event_id);
    } else if (name === 'confirm') {
      const goal = root.querySelector('#goal').value.trim();
      if (!goal) throw new Error('Enter a goal.');
      state = await api(`${base}/agents/${selected}/proposals/${proposals[selected].proposal_id}/confirm`, 'POST', { goal });
      proposals[selected] = null;
    } else if (name === 'respond') {
      const choice = root.querySelector('input[name="decision"]:checked')?.value;
      const reason = root.querySelector('#reason').value.trim();
      const confidence = root.querySelector('#confidence').value;
      const claim = root.querySelector('#claim')?.value.trim() || '';
      if (!choice || !reason || confidence === '' || state.phase === 'final_review' && !claim) throw new Error('Complete the choice, reason, confidence, and any requested claim.');
      state = await api(`${base}/responses`, 'POST', { choice_id: choice, reason, confidence: Number(confidence), claim_text: claim });
    } else if (name === 'correct') {
      const choice = root.querySelector('input[name="correction"]:checked')?.value;
      const reason = root.querySelector('#correction-reason').value.trim();
      if (!choice || !reason) throw new Error('Choose a corrected answer and give a reason.');
      state = await api(`${base}/corrections`, 'POST', { choice_id: choice, reason });
    }
  } catch (cause) { error = cause.message; }
  finally { busy = false; render(); }
}

function pcmWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
  write(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); write(8, 'WAVE'); write(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); write(36, 'data'); view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => view.setInt16(44 + index * 2, Math.max(-1, Math.min(1, sample)) * 0x7fff, true));
  return new Blob([buffer], { type: 'audio/wav' });
}

async function startRecording() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) { error = 'Microphone capture requires HTTPS and a compatible browser.'; render(); return; }
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) { error = 'Web Audio is unavailable.'; render(); return; }
  let ticket;
  let abandoned = false;
  recordingPurpose = state.phase === 'awaiting_briefing' ? 'briefing' : 'followup';
  recordingAgent = selected;
  try {
    recordingContext = recordingContext && recordingContext.state !== 'closed' ? recordingContext : new AudioContextClass();
    void recordingContext.resume();
    const permission = navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      if (abandoned) stream.getTracks().forEach((track) => track.stop());
      return stream;
    });
    ticket = api(`${base}/agents/${recordingAgent}/speech/activations?purpose=${recordingPurpose}`, 'POST');
    [recordingStream, activation] = await Promise.all([permission, ticket]);
    const source = recordingContext.createMediaStreamSource(recordingStream);
    const analyser = recordingContext.createAnalyser();
    const silent = recordingContext.createGain();
    silent.gain.value = 0.000001;
    analyser.fftSize = 2048;
    source.connect(analyser); analyser.connect(silent); silent.connect(recordingContext.destination);
    const chunks = [];
    const sample = () => { const data = new Float32Array(analyser.fftSize); analyser.getFloatTimeDomainData(data); chunks.push(data); };
    sample();
    const timer = setInterval(sample, Math.max(10, analyser.fftSize / recordingContext.sampleRate * 1000));
    recorder = { source, analyser, silent, timer, chunks, sampleRate: recordingContext.sampleRate };
    recording = true; error = ''; render();
  } catch (cause) {
    abandoned = true;
    error = cause.message;
    if (!activation && ticket) activation = await ticket.catch(() => null);
    if (activation) await api(`${base}/agents/${recordingAgent}/speech/activations/${activation.transcription_id}/abort`, 'POST').catch(() => {});
    recordingStream?.getTracks().forEach((track) => track.stop()); recordingStream = null; activation = null; render();
  }
}

async function stopRecording() {
  if (!recorder || !activation) return;
  recording = false; busy = true;
  render();
  const { source, analyser, silent, timer, chunks, sampleRate } = recorder;
  clearInterval(timer);
  const last = new Float32Array(analyser.fftSize); analyser.getFloatTimeDomainData(last); chunks.push(last);
  source.disconnect(); analyser.disconnect(); silent.disconnect();
  recordingStream?.getTracks().forEach((track) => track.stop()); recordingStream = null; recorder = null;
  const samples = new Float32Array(chunks.reduce((total, chunk) => total + chunk.length, 0));
  let offset = 0; for (const chunk of chunks) { samples.set(chunk, offset); offset += chunk.length; }
  try {
    const heard = await api(`${base}/agents/${recordingAgent}/speech?purpose=${recordingPurpose}&activation_id=${encodeURIComponent(activation.transcription_id)}`, 'POST', pcmWav(samples, sampleRate));
    if (recordingPurpose === 'briefing') {
      speech[recordingAgent] = heard;
      proposals[recordingAgent] = null;
    } else {
      followupSpeech[recordingAgent] = heard;
      followupResult[recordingAgent] = null;
      followupDraft[recordingAgent] = heard.recognized_transcript;
      if (heard.status === 'recognized') {
        followupResult[recordingAgent] = await api(`${base}/agents/${recordingAgent}/followups`, 'POST', {
          transcription_id: heard.transcription_id,
        });
        if (followupResult[recordingAgent].revealed_event_id) openedEvents.add(followupResult[recordingAgent].revealed_event_id);
      }
    }
    error = heard.status === 'recognized' ? '' : 'Speech was not recognized. Record again.';
  } catch (cause) { error = cause.message; }
  activation = null; busy = false; render();
}

let researcherToken = '';
let researcherForm = { participantId: '', slot: '', period: '1' };
const conditionCSlots = {
  '1': ['02', '04', '06', '08', '10', '12', '14'],
  '2': ['01', '03', '05', '07', '09', '11', '13', '15'],
};
const slotOptions = (period, chosen) => '<option value="">Select allocated slot</option>' +
  conditionCSlots[period].map((slot) => `<option value="${slot}" ${slot === chosen ? 'selected' : ''}>Slot ${slot}</option>`).join('');
function renderResearcher() {
  root.innerHTML = `<main><p class="label">Researcher setup</p><h1 class="title">Separate Agents</h1><p class="intro">Create a Condition C period using its allocated slot and period. The participant uses the generated link.</p>${error ? `<p class="error" role="alert">${esc(error)}</p>` : ''}<section class="card stack"><label>Researcher token<input id="researcher-token" type="password" value="${esc(researcherToken)}"></label><label>Participant ID<input id="participant-id" autocomplete="off" value="${esc(researcherForm.participantId)}"></label><label>Period<select id="period"><option value="1" ${researcherForm.period === '1' ? 'selected' : ''}>Period 1</option><option value="2" ${researcherForm.period === '2' ? 'selected' : ''}>Period 2</option></select></label><label>Counterbalancing slot<select id="slot">${slotOptions(researcherForm.period, researcherForm.slot)}</select></label><button class="button" id="create-trial">Create trial</button></section><section class="card stack"><h2>State probe</h2><label>Trial ID<input id="probe-trial"></label><label>Verbatim participant answer<textarea id="probe-answer"></textarea></label><button class="button secondary" id="save-probe">Save answer and resume</button><button class="button secondary" id="download-export">Download trial export</button></section><div id="result"></div></main>`;
  root.querySelector('#period').onchange = (event) => {
    researcherForm.period = event.target.value;
    researcherForm.slot = '';
    root.querySelector('#slot').innerHTML = slotOptions(researcherForm.period, '');
  };
  root.querySelector('#create-trial').onclick = async () => {
    researcherToken = root.querySelector('#researcher-token').value;
    researcherForm = {
      participantId: root.querySelector('#participant-id').value.trim(),
      slot: root.querySelector('#slot').value.trim(),
      period: root.querySelector('#period').value,
    };
    try {
      if (!/^[A-Za-z0-9_-]{1,80}$/.test(researcherForm.participantId)) throw new Error('Participant ID must contain only letters, digits, hyphens, or underscores.');
      if (!researcherForm.slot) throw new Error('Select a counterbalancing slot.');
      const result = await api('/study/researcher/trials', 'POST', {
        participant_id: researcherForm.participantId,
        slot: researcherForm.slot,
        period: Number(researcherForm.period),
      }, true);
      error = '';
      root.querySelector('.error')?.remove();
      root.querySelector('#result').innerHTML = `<section class="card"><h2>Participant link</h2><p><a href="${esc(result.participant_url)}">Open trial ${esc(result.trial.trial_id)}</a></p><p class="small">Copy this link to the study phone. Keep its access fragment private.</p></section>`;
      root.querySelector('#probe-trial').value = result.trial.trial_id;
    } catch (cause) { error = cause.message; renderResearcher(); }
  };
  root.querySelector('#save-probe').onclick = async () => {
    researcherToken = root.querySelector('#researcher-token').value;
    try {
      await api(`/study/researcher/trials/${root.querySelector('#probe-trial').value}/probe`, 'POST', { reason: root.querySelector('#probe-answer').value }, true);
      root.querySelector('#result').innerHTML = '<p class="notice">Probe saved. Trial resumed.</p>';
    } catch (cause) { error = cause.message; renderResearcher(); }
  };
  root.querySelector('#download-export').onclick = async () => {
    researcherToken = root.querySelector('#researcher-token').value;
    try {
      const id = root.querySelector('#probe-trial').value;
      const data = await api(`/study/researcher/trials/${id}/export`, 'GET', undefined, true);
      const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })); link.download = `${id}.json`; link.click(); URL.revokeObjectURL(link.href);
    } catch (cause) { error = cause.message; renderResearcher(); }
  };
}

if (trialId) {
  const key = `separate-access:${trialId}`;
  const fromLink = new URLSearchParams(location.hash.slice(1)).get('access');
  token = fromLink || sessionStorage.getItem(key) || '';
  if (fromLink) { sessionStorage.setItem(key, fromLink); history.replaceState(null, '', location.pathname); }
  selected = sessionStorage.getItem(`separate-agent:${trialId}`) || '';
  if (!token) error = 'This trial link is missing its access code. Ask the researcher for the participant link.';
  render(); void refresh(); setInterval(refresh, 1500);
} else renderResearcher();
