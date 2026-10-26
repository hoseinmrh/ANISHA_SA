# Frozen wine scenario stimuli — condition-c-0.6

Use one variant per condition according to `counterbalancing.csv`. The study runner displays the same project facts and source records in Full ANISHA and Condition C; their voice-entry structure and organisation differ. Delegation starts with natural speech in both modes. The same speech-recognition service creates a reviewable transcript, and a deterministic mapper converts the participant-approved wording to the fixed Coding and Writing proposal. The researcher never paraphrases a source rule or hints at a correct answer. This is simulated project information, not a real analysis of the wine dataset.

## Common participant instruction before either variant

> You are supervising two AI agents on a small university wine-quality analysis. Coding checks the data and calculates results. Writing drafts the report; its results section must wait for verified Coding output. You are responsible for resolving conflicting instructions, reviewing actions that need approval, and deciding whether the final report is ready. While they work, scan the paper sheet and mark every exact `Q7` token from left to right. A cue means you may finish your current token, then check the phone. Return to the next token in the same row afterward. The interface is being evaluated; your performance is not graded.

Training establishes: an association does not prove causation; if the project says “significant only when p < .05,” a p value of .18 is not significant. Do not mention either wine variant's injected claim during training.

### Neutral training project (same for everyone)

> Project: Campus library open-house note. Ask Coding to count entries in a provided practice table. Ask Writing to draft a two-sentence announcement using the verified count. Coding's count must be checked before Writing inserts it. A sample status says Coding is working; a sample cue says Writing needs a confirmed count. Open the source table, choose “request correction” on a deliberately wrong practice count, then reject a practice request to delete the source table. Return to a three-row practice `Q7` sheet. Nothing in this practice project is scored or kept as study data.

The practice source table contains three entries: Monday, Tuesday, Wednesday. The intentionally wrong practice status says “Coding counted four entries”; the correct practice count is three. The practice deletion request says “Delete the original source table after counting.” The instruction says to preserve the original table. These practice facts are unrelated to either wine variant.

Practice visual-search rows (print separately; `Q7` is the only target):

```text
01  O7  Q1  Q7  P7  QZ  O1  Q2  Q7  R7  7Q
02  Q2  O7  P1  Q9  Q1  QZ  Q7  0Q  P7  O1
03  QZ  Q7  O7  7Q  P1  Q2  O1  R7  Q9  Q7
```

Researcher says: “In a separate example, a survey finds coffee use and short sleep occur together. Does that observation by itself prove coffee caused the short sleep?” Explain the answer is no. Then say: “If a report uses p < .05 as its rule, is p = .20 below that threshold?” Explain the answer is no. These examples teach how to read a rule without revealing the wine answers. The practice rows above are discarded before the measured sheets begin.

## Shared source and action rules

Source precedence is explicit in both interfaces: a dated latest supervisor instruction outranks an older draft note. Coding results must be verified before Writing can fill the results section. The participant may open source records, history, and pending actions in either interface. The researcher freezes all text, timestamps, and outputs before the main study.

### Voice delegation in each interface

**Full ANISHA:** Show one shared microphone control for both agents and for later status/evidence/action commands. Researcher says: “Using this one voice entry, tell ANISHA what you want both Coding and Writing to do, including any constraints you think matter. Review the transcript and combined proposal before confirming.” The participant may re-speak or correct a recognition error; log every activation/repair. A successful first entry covers both agents. The interface may ask a fixed clarification if a task or constraint is missing, but it does not dispatch until the participant explicitly confirms the combined proposal.

**Condition C:** Show separate Coding and Writing screens, each with the same microphone control and recogniser; later voice commands apply to the currently selected agent. Researcher says: “Open each agent and brief it by voice. Review and confirm each transcript and proposal before the work starts.” The participant chooses the order and can re-speak or repair a recognition error. There is no shared voice entry or combined proposal. The runner starts the common trace only after both agent briefs have been confirmed.

The experiment does not require a verbatim phrase: participant wording, activation count, recognition errors, and corrections are genuine behaviour. The proposal maps to the same frozen tasks and constraints in either condition after confirmation. Do not substitute a typed delegation in a measured session. If speech recognition or mapping prevents valid dispatch, log a technical deviation and use the predefined fixed proposal only with the participant's confirmation; flag the trial for review under the analysis plan.

### Variant 1: project brief and evidence cards

**Participant brief (show before delegation):**

> Project: Wine-quality report, version 1. Ask Coding to validate the dataset and examine how alcohol and volatile acidity are associated with quality. Ask Writing to draft a report structure and wait for verified Coding results before writing the results section. Preserve the original report; create a review draft. Describe associations, not causes. Do not publish or merge anything without review.

**Source V1-S1 — latest supervisor instruction, 17 October 2026, authoritative:**

> For this analysis, high quality means a quality score of 7 or above. Preserve the original report. Report observed associations only; the analysis does not establish causation.

**Source V1-S2 — earlier working note, 2 October 2026, superseded:**

> For an exploratory chart, high quality was defined as above this sample's median. This was a draft definition and has not been reconfirmed.

**Source V1-S3 — verified analysis summary (released at Coding completion):**

> Coding completed the planned descriptive analysis using score ≥ 7 as the high-quality rule. The checked output reports associations between alcohol, volatile acidity, and quality. No causal design or intervention was used.

**Variant 1 Writing draft preview (available with approval request):**

> Wine-quality report — working review draft.
> Aim: Describe observed relationships of alcohol and volatile acidity with wine quality.
> Structure: Project aim; dataset validation and descriptive method; results; discussion.
> Results: Waiting for verified Coding output. The results and discussion text are not complete in this draft.

### Variant 2: project brief and evidence cards

**Participant brief (show before delegation):**

> Project: Wine-quality report, version 2. Ask Coding to validate the dataset and assess whether verified extreme alcohol or volatile-acidity values change the reported relationship. Ask Writing to draft a report structure and wait for verified Coding results before writing the results section. Keep a new version rather than overwriting an original. Do not publish before review. Call a result statistically significant only when p < .05.

**Source V2-S1 — latest supervisor instruction, 17 October 2026, authoritative:**

> Retain plausible extreme alcohol and acidity values after verification; do not trim them automatically. Make a new report version. Nothing is published before review. Use p < .05 as the significance rule.

**Source V2-S2 — earlier working note, 2 October 2026, superseded:**

> For a preliminary clean-up, remove the top and bottom one percent before analysis. This draft rule has not been reconfirmed.

**Source V2-S3 — verified analysis summary (released at Coding completion):**

> Coding verified that the extreme values are plausible and retained them in the planned analysis. The reported test for volatile acidity has p = .18. The project rule is p < .05.

**Variant 2 Writing draft preview (available with approval request):**

> Wine-quality report — working review draft.
> Aim: Describe how verified extreme alcohol and volatile-acidity values affect the reported relationship with quality.
> Structure: Project aim; dataset validation and analysis method; results; discussion.
> Results: Waiting for verified Coding output. The results and discussion text are not complete in this draft.

### Fixed final review files

At the Coding-completion event, release the matching verified summary (V1-S3
or V2-S3) and Coding script from `final_artifacts/` in Coding's conversation.
Variant 1 uses `wine_1_analysis.py`; variant 2 uses `wine_2_analysis.py`.
Writing's results section remains waiting for the verified Coding result.
At final review, release the matching Writing Markdown report:
`wine_1_report.md` or `wine_2_report.md`. These are fixed study review files,
not live agent output. The Markdown report contains the exact final report
excerpt below. Released files remain visible after the final answer. File
contents are included in the scenario checksum recorded with each trial.

## Fixed event copy

The event IDs and timing for this Condition C version are in `event_trace.csv`.
Routine events update the relevant status screen silently. At the blocker pair,
use one device cue for the two co-present items. The 45-second offset is from
the probe response to that pair. After the first blocker response and fixed
feedback, scripted Coding completes and its verified result is released at a
zero-second offset with one Coding cue. There is no post-blocker wait. The
approval request is due no earlier than 90 seconds after the blocker response
and only after Writing receives a valid participant send. No result is inserted
into Writing automatically.

| Script key | Variant 1 copy | Variant 2 copy |
| --- | --- | --- |
| `tasks_accepted` | “Coding: validate data and analyse alcohol/acidity associations. Writing: draft the report structure; wait for verified Coding results. Review draft only.” | “Coding: validate data and test verified extremes. Writing: draft the report structure; wait for verified Coding results. New version; review before publication.” |
| `validation_progress` | “Coding: dataset opened; validation in progress.” | Same text. |
| `outline_complete` | “Writing: report outline complete. Results section waiting for verified Coding output.” | Same text. |
| `analysis_started` | “Coding: validation passed; analysis in progress.” | Same text. |
| `probe_dependency` | Display blanked; researcher asks the exact state question in the questionnaire appendix of `Deliverables/ANISHA_Study_Plan.tex`. | Same procedure and question. |
| `conflicting_definition` / `conflicting_outlier_rule` | “Coding blocked. The old note says high quality is above the sample median; the latest supervisor instruction says score ≥ 7. Which rule should Coding use? You may inspect both sources.” | “Coding blocked. The old note says remove the top and bottom one percent; the latest supervisor instruction says retain verified plausible extremes. Which rule should Coding use? You may inspect both sources.” |
| `competing_completion` | “Writing: title page and formatting pass complete. Results still waiting for Coding.” | Same text. |
| `coding_complete` | “Coding: the analysis is complete and verified. Open V1-S3 and the Coding script to review the result. Writing's results section is waiting for this verified result.” | “Coding: the analysis is complete and verified. Open V2-S3 and the Coding script to review the result. Writing's results section is waiting for this verified result.” |
| `writing_requests_result` | “I have finished the outline. Please open Coding's verified analysis result and send it to me here so I can finish the results section.” | Same text. |
| `overwrite_request` / `publish_request` | “Approval requested: overwrite the original report with this draft. Approve, reject, or defer? Give a brief reason.” | “Approval requested: publish the draft summary before review. Approve, reject, or defer? Give a brief reason.” |
| `final_review` | “Final review: see the report excerpt and verified analysis source. Identify any claim requiring correction and decide whether the report is ready to accept. Give a brief reason.” | Same instruction. |

**Variant 1 final report excerpt:**

> The analysis grouped wines with quality score 7 or above as high quality. Alcohol and volatile acidity were associated with quality in the checked descriptive output. **Higher alcohol causes higher wine quality.** The report has not been published.

**Variant 2 final report excerpt:**

> The verified extreme values were retained. The checked output reports p = .18 for volatile acidity. **Volatile acidity is statistically significant.** The report has not been published.

## Fixed branch feedback and safety

At each gate, save the participant's first response and their confidence before feedback. The researcher must not restate the right answer before the first response is captured.

- If blocker answer is correct: “The analysis will continue using the selected rule.”
- If blocker answer is wrong or unclear: “The selected rule conflicts with the latest supervisor instruction. You may correct your answer once.” Save correction separately, then continue the same trace using the latest rule regardless of the correction.
- If approval is rejected: “The prohibited action was not executed. Work continues as a review draft.”
- If approval is approved or deferred: after saving first choice and confidence, show “This request conflicts with a project constraint. No external action was executed. You may correct your answer once.” Save any correction separately. Deferral is safe but incomplete for the first-response rubric because the stated constraint already rules out the action.
- After final answer: do not confirm correctness until the post-condition questionnaires and both conditions are finished.

For the Condition C handoff, Writing accepts the complete verified V1-S3 or
V2-S3 summary text. The first and subsequent submitted payloads are preserved.
An incomplete or different payload receives the fixed clarification below. A
valid send receives the fixed receipt below. No result appears in Writing's
conversation until the participant sends it. There is no automatic transfer or
timeout transfer in this version.

- If handoff send is invalid: “I still need Coding's verified analysis result. Please check the verified source in Coding and send the complete result here.”
- If handoff send is valid: “I received Coding's verified analysis result. I can continue the results section.”

No actual file overwrite, publication, merge, package installation, or network side effect is possible in any branch. The fixed final artefact appears in every branch, so later task difficulty does not depend on earlier correctness.

## Interface parity acceptance test

The researcher must locate V1-S1/S2/S3 and V2-S1/S2/S3, blocker choices, approval choices, final excerpt, and evidence/history in both modes. The screens may organise these differently, but may not omit facts or require impossible navigation. Both modes use the same phone microphone, speech-recognition service/version, language, transcript review, and explicit confirmation. ANISHA has one combined voice entry; Condition C requires separate agent briefings. Any spoken event summaries must use the same frozen wording and cue policy in both modes. Test every branch with a scripted wrong answer and ensure feedback and subsequent events are identical.
