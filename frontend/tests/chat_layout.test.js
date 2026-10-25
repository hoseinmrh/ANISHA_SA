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

async function loadAgent(agent) {
  const root = { innerHTML: '', querySelectorAll: () => [], querySelector: () => null };
  const interactions = [];
  const context = vm.createContext({
    document: { querySelector: () => root, addEventListener: () => {} },
    location: { pathname: '/trial/study-layout', hash: '#access=smoke' },
    sessionStorage: { getItem: key => key.includes('separate-agent') ? agent : null, setItem: () => {} },
    history: { replaceState: () => {} }, URLSearchParams, Blob, setInterval: () => 0,
    fetch: async (url, options = {}) => {
      if (url.endsWith('/state')) return { ok: true, json: async () => trialState() };
      if (url.endsWith('/interactions')) {
        interactions.push(JSON.parse(options.body));
        return { ok: true, json: async () => ({ is_new: true }) };
      }
      throw new Error(`Unexpected request: ${url}`);
    },
  });
  vm.runInContext(script, context);
  await new Promise(resolve => setTimeout(resolve, 0));
  return { root, context, interactions };
}

test('each agent keeps references and activity outside its document chat', async () => {
  for (const [agent, ownFile, otherFile, ownUpdate, otherUpdate] of [
    ['coding_agent', 'Coding analysis.py', 'Writing report.md', 'Coding status update text', 'Writing status update text'],
    ['writing_agent', 'Writing report.md', 'Coding analysis.py', 'Writing status update text', 'Coding status update text'],
  ]) {
    const { root, context, interactions } = await loadAgent(agent);
    assert.match(root.innerHTML, new RegExp(ownFile.replace('.', '\\.')));
    assert.doesNotMatch(root.innerHTML, new RegExp(otherFile.replace('.', '\\.')));
    assert.doesNotMatch(root.innerHTML, /Original project briefing text|Supervisor reference text|status update text/);

    await vm.runInContext("action({ currentTarget: { dataset: { action: 'details', panel: 'context' } } })", context);
    assert.match(root.innerHTML, /Original project briefing text/);
    assert.match(root.innerHTML, /Supervisor note, 17 October 2026/);
    assert.doesNotMatch(root.innerHTML, /Supervisor reference text/);

    await vm.runInContext("action({ currentTarget: { dataset: { action: 'source', id: 'W1-S1' }, closest: () => null } })", context);
    assert.match(root.innerHTML, /Supervisor reference text/);

    await vm.runInContext("action({ currentTarget: { dataset: { action: 'details', panel: 'activity' } } })", context);
    assert.ok(root.innerHTML.includes(ownUpdate));
    assert.ok(!root.innerHTML.includes(otherUpdate));
    assert.deepEqual(interactions.map(item => item.event_type), ['ui_layer_open', 'source_open', 'ui_layer_open']);
  }
});
