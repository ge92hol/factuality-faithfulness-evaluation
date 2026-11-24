#!/bin/bash
# Clone the upstream evaluator repositories into external/ at the commits used for the results.
# Run from the repository root. external/ is git-ignored (third-party code, ~240 MB).
#   Q-CaRE-COLM-26  imported at runtime by the Q-CARE adapter (prompts, pipeline, metrics)
#   ConflictScore   label parser ported into the adapter; used by the tests
#   RAGChecker      reference copy of the installed ragchecker==0.1.9 package
#   RA-RAG, llm-uncertainty-perceptions, metametrics  reviewed in the spec; provenance only
set -euo pipefail
mkdir -p external
clone() {  # name url sha
  if [ ! -d "external/$1/.git" ]; then git clone --quiet "$2" "external/$1"; fi
  git -C "external/$1" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "external/$1" checkout --quiet "$3"
  echo "$1 $(git -C "external/$1" rev-parse HEAD)"
}
clone ConflictScore               https://github.com/microsoft/ConflictScore.git              a8868e5cd98bff81935a1766b54a02ad3eab9e8f
clone Q-CaRE-COLM-26              https://github.com/DISL-Lab/Q-CaRE-COLM-26.git              e8b48e12a28f308076ffeae122cd8c765717a6b7
clone RAGChecker                  https://github.com/amazon-science/RAGChecker.git            6091f08c00e676e87a970f2aeb4a23a484746348
clone RA-RAG                      https://github.com/ml-postech/RA-RAG.git                    7479bb04cdee4ee48be643987749166a517302b5
clone llm-uncertainty-perceptions https://github.com/UCIDataLab/llm-uncertainty-perceptions.git 553b34b80238b4a1f3698b6bf83f46e399efc4d9
clone metametrics                 https://github.com/meta-metrics/metametrics.git             0bcda3f78895efb87653b9bf2f23086416d7cd52
