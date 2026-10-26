const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const script = fs.readFileSync(path.join(__dirname, '..', 'app.js'), 'utf8');

function trialState() {
  return {
    trial_id: 'study-layout', interface_label: 'Interface A', phase: 'running',
    participant_brief: 'Original project briefing text', masked: false,
    confirmed_agents: ['coding_agent', 'writing_agent'],
    tasks: [
      { task_id: 'study-layout:coding', agent_id: 'coding_agent', goal: 'Check analysis', state: 'running', depends_on: null },
      { task_id: 'study-layout:writing', agent_id: 'writing_agent', goal: 'Draft report', state: 'running', depends_on: null },
    ],
    records: [
      { record_id: 'common-instruction', kind: 'instruction', title: 'Study instructions', text: 'Shared instruction text' },
      { record_id: 'W1-S1', kind: 'source', title: 'Supervisor note, 17 October 2026', text: 'Supervisor reference text' },
      { record_id: 'final-code', kind: 'code_artifact', title: 'Coding analysis.py', text: 'print("analysis")' },
      { record_id: 'final-markdown', kind: 'markdown_artifact', title: 'Writing report.md', text: '# New report' },
    ],
    events: [
      { event_id: 'C1', task_id: 'study-layout:coding', text: 'Coding status update text', occurred_at_ms: 1000000, cue_id: null },
      { event_id: 'W1', task_id: 'study-layout:writing', text: 'Writing status update text', occurred_at_ms: 1001000, cue_id: null },
    ],
    action_options: [], correction_options: [],
  };
}

async function loadAgent(agent, currentState = trialState()) {
  const root = { innerHTML: '', handoffValue: '', querySelectorAll: () => [],
    querySelector: selector => selector === '#handoff-payload' ? { value: root.handoffValue, addEventListener: () => {} } : null };
  const interactions = [];
  const copiedTexts = [];
  const handoffPayloads = [];
  const context = vm.createContext({
    document: { querySelector: () => root, addEventListener: () => {} },
    location: { pathname: '/trial/study-layout', hash: '#access=smoke' },
    navigator: { vibrate: () => {}, clipboard: { writeText: async text => { copiedTexts.push(text); } } },
    sessionStorage: { getItem: key => key.includes('separate-agent') ? agent : null, setItem: () => {} },
    history: { replaceState: () => {} }, URLSearchParams, Blob, setInterval: () => 0,
    fetch: async (url, options = {}) => {
      if (url.endsWith('/state')) return { ok: true, json: async () => currentState };
      if (url.endsWith('/interactions')) {
        interactions.push(JSON.parse(options.body));
        return { ok: true, json: async () => ({ is_new: true }) };
      }
      if (url.endsWith('/handoff')) {
        handoffPayloads.push(JSON.parse(options.body).payload);
        return { ok: true, json: async () => currentState };
      }
      throw new Error(`Unexpected request: ${url}`);
    },
  });
  vm.runInContext(script, context);
  await new Promise(resolve => setTimeout(resolve, 0));
  return { root, context, interactions, copiedTexts, handoffPayloads };
}

test('each agent shows its own updates in chat and keeps shared references in Context', async () => {
  for (const [agent, ownFile, otherFile, ownUpdate, otherUpdate] of [
    ['coding_agent', 'Coding analysis.py', 'Writing report.md', 'Coding status update text', 'Writing status update text'],
    ['writing_agent', 'Writing report.md', 'Coding analysis.py', 'Writing status update text', 'Coding status update text'],
  ]) {
    const { root, context, interactions } = await loadAgent(agent);
    assert.match(root.innerHTML, new RegExp(ownFile.replace('.', '\\.')));
    assert.doesNotMatch(root.innerHTML, new RegExp(otherFile.replace('.', '\\.')));
    assert.match(root.innerHTML, new RegExp(ownUpdate));
    assert.doesNotMatch(root.innerHTML, new RegExp(otherUpdate));
    assert.doesNotMatch(root.innerHTML, /Original project briefing text|Supervisor reference text|data-panel="activity"/);

    await vm.runInContext("action({ currentTarget: { dataset: { action: 'details', panel: 'context' } } })", context);
    assert.match(root.innerHTML, /Original project briefing text/);
    assert.match(root.innerHTML, /Supervisor note, 17 October 2026/);
    assert.doesNotMatch(root.innerHTML, /Supervisor reference text/);

    await vm.runInContext("action({ currentTarget: { dataset: { action: 'source', id: 'W1-S1' }, closest: () => null } })", context);
    assert.match(root.innerHTML, /Supervisor reference text/);

    assert.deepEqual(interactions.map(item => item.event_type), ['ui_layer_open', 'source_open']);
  }
});

test('each agent renders its saved chat without the other agent\'s messages', async () => {
  const currentState = trialState();
  currentState.conversation = [
    { id: 'c-brief', agent_id: 'coding_agent', kind: 'briefing', transcript: 'Check the data', goal: 'Analyze wine data' },
    { id: 'w-brief', agent_id: 'writing_agent', kind: 'briefing', transcript: 'Draft the report', goal: 'Write a wine report' },
    { id: 'c-q', agent_id: 'coding_agent', kind: 'followup', transcription_id: 'c-speech', transcript: 'How is the analysis?', message: 'Coding is running.', record_ids: [] },
    { id: 'w-q', agent_id: 'writing_agent', kind: 'followup', transcription_id: 'w-speech', transcript: 'How is the report?', message: 'Writing is running.', record_ids: [] },
  ];
  const coding = await loadAgent('coding_agent', currentState);
  assert.match(coding.root.innerHTML, /Check the data/);
  assert.match(coding.root.innerHTML, /How is the analysis\?/);
  assert.doesNotMatch(coding.root.innerHTML, /Draft the report|How is the report\?/);
  const writing = await loadAgent('writing_agent', currentState);
  assert.match(writing.root.innerHTML, /Draft the report/);
  assert.match(writing.root.innerHTML, /How is the report\?/);
  assert.doesNotMatch(writing.root.innerHTML, /Check the data|How is the analysis\?/);
});

test('a refresh restores an unfinished briefing in the selected chat', async () => {
  const currentState = trialState();
  currentState.phase = 'awaiting_briefing';
  currentState.confirmed_agents = [];
  currentState.conversation = [];
  currentState.pending_briefings = {
    coding_agent: {
      speech: { transcription_id: 'speech-1', activation_index: 1, recognized_transcript: 'Analyze the wine', status: 'recognized' },
      proposal: { proposal_id: 'proposal-1', reviewed_transcript: 'Analyze the wine', goal: 'Analyze the wine', constraints: [], source_refs: [], approval_boundary: 'Review before publishing.' },
    },
  };
  const { root } = await loadAgent('coding_agent', currentState);
  assert.match(root.innerHTML, /Analyze the wine/);
  assert.match(root.innerHTML, /Confirm Coding briefing/);
});

test('verified Coding summary appears only in Coding conversation', async () => {
  const currentState = trialState();
  currentState.records.push({
    record_id: 'V1-S3', kind: 'source',
    title: 'verified analysis summary (released at Coding completion)',
    text: 'Checked descriptive associations',
  });
  const coding = await loadAgent('coding_agent', currentState);
  assert.match(coding.root.innerHTML, /Verified analysis summary/);
  assert.match(coding.root.innerHTML, /Checked descriptive associations/);
  const writing = await loadAgent('writing_agent', currentState);
  assert.doesNotMatch(writing.root.innerHTML, /Checked descriptive associations|V1-S3/);
});

test('chat orders voice, updates, decisions and released files by saved time', async () => {
  const currentState = trialState();
  currentState.conversation = [
    { id: 'brief', agent_id: 'coding_agent', kind: 'briefing', transcript: 'Check the data', goal: 'Analyze wine', created_at_ms: 900000 },
    { id: 'question', agent_id: 'coding_agent', kind: 'followup', transcript: 'What changed?', message: 'I checked the rule.', record_ids: [], created_at_ms: 1001500 },
  ];
  currentState.events = [
    { event_id: 'C1', task_id: 'study-layout:coding', text: 'Validation began', occurred_at_ms: 1000000, cue_id: null },
    { event_id: 'C2', task_id: 'study-layout:coding', text: 'Blocker feedback', occurred_at_ms: 1002000, cue_id: null },
    { event_id: 'C3', task_id: 'study-layout:coding', text: 'Coding complete', occurred_at_ms: 1003000, cue_id: null },
  ];
  currentState.decisions = [
    { response_id: 'answer', gate_event_id: 'C2', task_id: 'study-layout:coding', response_order: 1,
      choice_label: 'Use rule from V1-S1', reason: 'Latest note', claim_text: '', responded_at_ms: 1002000 },
  ];
  currentState.records = [
    { record_id: 'W1-S1', kind: 'source', title: 'Supervisor note', text: 'Keep the source' },
    { record_id: 'V1-S3', kind: 'source', title: 'Verified result', text: 'Checked result', release_event_id: 'C3' },
    { record_id: 'final-code', kind: 'code_artifact', title: 'Coding analysis.py', text: 'print(1)', release_event_id: 'C3' },
  ];
  const { root } = await loadAgent('coding_agent', currentState);
  const expected = ['Check the data', 'Validation began', 'What changed?', 'Use rule from V1-S1',
    'Blocker feedback', 'Coding complete', 'Verified analysis summary', 'Coding analysis.py'];
  let previous = -1;
  for (const phrase of expected) {
    const at = root.innerHTML.indexOf(phrase);
    assert.ok(at > previous, `${phrase} must appear after the preceding timeline item`);
    previous = at;
  }
  assert.doesNotMatch(root.innerHTML, /Keep the source/);
});

test('cued update stays unread in chat until opened', async () => {
  const currentState = trialState();
  currentState.events = [{ event_id: 'C-cue', task_id: 'study-layout:coding', text: 'Coding needs review',
    occurred_at_ms: 1000000, cue_id: 'one-cue' }];
  const { root, context, interactions } = await loadAgent('coding_agent', currentState);
  assert.match(root.innerHTML, /Coding <span class="badge">1<\/span>/);
  assert.match(root.innerHTML, /New update ready to open/);
  assert.doesNotMatch(root.innerHTML, /Coding needs review/);
  await vm.runInContext("action({ currentTarget: { dataset: { action: 'event', id: 'C-cue' } } })", context);
  assert.match(root.innerHTML, /Coding needs review/);
  assert.doesNotMatch(root.innerHTML, /Coding <span class="badge">1<\/span>/);
  assert.deepEqual(interactions.map(item => item.event_type), ['cue_delivered', 'event_open']);
});

test('participant copies in Coding and explicitly pastes and sends in Writing', async () => {
  const currentState = trialState();
  currentState.phase = 'handoff';
  currentState.events.push({ event_id: 'W-request', task_id: 'study-layout:writing',
    text: "Please open Coding's verified analysis result and send it to me here.",
    occurred_at_ms: 1002000, cue_id: null });
  currentState.records.push({ record_id: 'V1-S3', kind: 'source', title: 'Verified analysis',
    text: 'Complete verified result', release_event_id: 'C1' });
  const coding = await loadAgent('coding_agent', currentState);
  assert.match(coding.root.innerHTML, /Copy verified result/);
  await vm.runInContext("action({ currentTarget: { dataset: { action: 'copy-result', id: 'V1-S3' } } })", coding.context);
  assert.deepEqual(coding.copiedTexts, ['Complete verified result']);
  assert.ok(coding.interactions.some(item => item.event_type === 'result_copy' && item.target_id === 'V1-S3'));
  const writing = await loadAgent('writing_agent', currentState);
  assert.match(writing.root.innerHTML, /Please open Coding&#39;s verified analysis result/);
  assert.match(writing.root.innerHTML, /id="handoff-payload"[^>]*><\/textarea>/);
  assert.doesNotMatch(writing.root.innerHTML, /Copy verified result|Complete verified result/);
  writing.root.handoffValue = 'Complete verified result';
  await vm.runInContext("action({ currentTarget: { dataset: { action: 'send-handoff' }, disabled: false } })", writing.context);
  assert.deepEqual(writing.handoffPayloads, ['Complete verified result']);
});
