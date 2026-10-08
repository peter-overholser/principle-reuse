# Clean program: deviations and run record

Protocol: `CP_PROTOCOL.md`. Registration: `cp_registration.json` (5 October
2026). Written 8 October 2026, after the registered analysis.

## Run record

- **Training, continuation and freeze** ran on the GB10 as one chain, from
  5 October to the night of 6–7 October. **The run was not interrupted or
  restarted** (reported by the investigator, 8 October).
- **Freeze** (`cp_freeze.json`) was pulled to the Mac and kept before the
  secret (`.cp_reveal_main`) was sent to the GB10, as the protocol requires.
- **Evaluation, reach and analysis** ran on the GB10 as one chain on
  7 October: 2,016 of 2,016 runs evaluated and 1,224 of 1,224 reach files
  written, none failed. The analysis ran once, at the end of that chain,
  with status "confirmatory".
- **Fingerprints recorded by the analysis** (`cp_analysis.json`,
  `integrity`): freeze `e001d8a2…`, outcomes `b69d74d4…`, reach
  `02a22eca…`.

## Deviations

1. **Outcome files reached the Mac before the analysis had run.**
   On 7 October, while the reach stage was still running, the step-10 pull
   was made early. It copied `outcomes_cp/` and the reach files written so
   far to the Mac; the report and `cp_analysis.json` did not yet exist.
   The copied files were not opened or analysed on the Mac. The registered
   analysis ran afterwards on the GB10, from the GB10's own files, by the
   registered code. The pull was repeated after the analysis finished.
   *Effect on the result: none. The analysis and its inputs were fixed by
   the freeze and the registered code before any outcome was seen.*

No other deviations are known.

## Not deviations, recorded for completeness

- **The A3 amendment** (transferred-probe *R*² in place of CKA) was made on
  3 October, before registration, and is part of the registered protocol.
- **The exploratory analyses** (protocol §8) use `explore_cp.py`, which is
  not among the registered code files. They run after the registered
  analysis, make no claims, and are reported separately in
  `CP_EXPLORATORY.md`.
