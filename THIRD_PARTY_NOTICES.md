# Third-party notices

## ConflictScore (Microsoft)

`src/benchmark/evaluators/conflictscore_published.py` contains `parse_label`, a port of the label-parsing
code in `src/eval_with_conflict_score/eval_all_calibration.py` (lines 104-140) of
<https://github.com/microsoft/ConflictScore> at commit `a8868e5cd98bff81935a1766b54a02ad3eab9e8f`.
The prompts in that module are transcribed from the ConflictScore paper (Liu et al., EMNLP 2026,
Appendix E, Tables 10-11). That code is used under the following license:

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

## Other upstream projects

RAGChecker (`ragchecker`, `refchecker`; Apache-2.0) is installed as a Python dependency, and Q-CARE (MIT, KAIST)
(<https://github.com/DISL-Lab/Q-CaRE-COLM-26>) is imported at runtime from `external/`.
Neither is redistributed in this repository; their licenses are in the respective packages and repositories.
