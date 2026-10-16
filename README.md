# Support Program Similarity

지원사업 문서를 비교하는 **Qwen3-Embedding-8B + Qwen3-Reranker-0.6B** 실행 코드입니다. 준비된 사업별 본문을 입력받아 코사인 검색 후 상위 20개를 재순위화합니다.

## 선정 기준과 검증 범위

검증 표본의 확대·전이 시험 이력을 먼저 고려하고, 같은 조건에서 모든 알려진 정답을 회수하는 데 필요한 최대 순위(K), MAP 순으로 판단했습니다. 작은 표본의 최고 점수를 대표 성적으로 사용하지 않습니다.

통합 실험은 질의 13개(기존 6개 + 추가 7개), 후보 140개, 알려진 정답 관계 19개입니다. 채택 구성의 MAP은 **0.567435**, MRR은 **0.586993**, **Recall@27=1.0**입니다. 저장 벡터와 reranker 점수로 전체 순위·지표를 재계산해 기존 결과와 일치함을 확인했습니다. 이후 게시 검토에서 토큰 디코딩으로 4개 문서의 문자가 달라지는 문제를 발견해 원문 위치 기반 분할로 수정했습니다. 153개 원문 모두 보존되는 것을 확인하고 변경된 4개 문서만 GPU 재계산했습니다. 3개 질의의 일부 후보 순서는 바뀌었지만 알려진 정답 기준 지표는 동일했습니다. 상위 20개 후보 집합은 유지되어 reranker 점수를 재사용했습니다.

Qwen+LCS+Reranker는 같은 통합 표본에서 K=50으로 악화했습니다. 수혜대상·조건만 추출한 방법은 K=25였지만 역방향 Recall@10이 0.7368에서 0.5789로 하락했습니다. 목적·내용 추출도 역방향 Recall@10이 0.6842로 하락했습니다. 이 공개판은 전체 입력 기반의 검색·재순위화 구성을 선택합니다. 모든 지표에서 우월한 구성이라는 뜻은 아닙니다. 채택 구성도 코사인 단독보다 Recall@10이 낮습니다(0.8077 → 0.6923).

Qwen Reranker는 별도 기존 10개 역방향 질의 시험에서 K=20→8 개선 이력이 있습니다. 그 시험은 후보 풀과 첫 검색 조건이 다르므로 현재 통합 성적과 합산하지 않습니다. 기존 문서·정답을 재사용한 전이 시험이며 **독립 미사용 사업에 대한 일반화 검증은 아직 없습니다**.

## 실행

Python 3.10 이상. 추론에는 로컬 모델 스냅샷 두 개와 충분한 메모리가 필요합니다. 모델 다운로드와 가중치는 포함하지 않습니다.

```sh
python -m pip install -r requirements-inference.txt
python -m support_program_similarity.retrieval --corpus /data/programs.json --model /models/Qwen3-Embedding-8B --reranker /models/Qwen3-Reranker-0.6B --output /runs/new-run
```

입력 JSON은 다음 형식입니다. id는 전체에서 고유해야 합니다. excluded_candidate_ids는 질의의 선택 항목이며, 동일 원문과 동일 ID 후보도 제외합니다.

```json
[
  {"id":"Q1","role":"query","text":"질의 사업 본문","excluded_candidate_ids":[]},
  {"id":"C1","role":"candidate","text":"후보 사업 본문"}
]
```

원문 문자를 보존하면서 최대 4096 토큰 단위로 입력을 나눠 정규화 임베딩을 만들고, 토큰 수로 가중 평균한 뒤 다시 L2 정규화합니다. 질의에만 검색 지시문을 붙입니다. 코사인 상위 20개는 Qwen Reranker의 yes/no 로짓 점수로 재정렬하고 나머지 순위는 유지합니다. reranker 입력이 32768 토큰을 넘으면 오류로 종료합니다. 출력 폴더가 이미 존재해도 덮어쓰지 않고 종료합니다.

권장 reranker 스냅샷은 Qwen/Qwen3-Reranker-0.6B의 e61197ed45024b0ed8a2d74b80b4d909f1255473입니다. 실행 환경 참고 버전은 NumPy 2.2.6, PyTorch 2.8.0+cu128, Transformers 4.57.1, Sentence Transformers 5.1.2입니다. GPU·라이브러리·가중치 변경 시 동일 수치를 보장하지 않습니다.

## 오프라인 평가

추론 프로세스는 정답 파일을 읽지 않습니다. 아래 평가 명령은 별도로 실행합니다. 파일 접근 격리를 강제하는 OS 샌드박스는 이 코드에 포함되어 있지 않습니다.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m support_program_similarity --rankings examples/rankings.synthetic.json --positives examples/positives.synthetic.json --k 3
```

평가기 입력 두 파일은 각각 query_id -> candidate_id 목록 JSON입니다. AP 분모는 누락된 정답을 포함한 전체 알려진 정답 수입니다. MAP@K, MRR@K, 질의별 Recall@K의 매크로 평균을 계산합니다. all_positive_min_k는 모든 정답이 K 안에 있을 때 필요한 최대 순위이고, 하나라도 빠지면 null입니다. 정답이 없는 질의는 별도 평가가 필요하며 이 평가기는 오류로 처리합니다. 미판정 후보를 확정 음성으로 해석해서는 안 됩니다.

## 공개 범위

이 저장소는 준비된 사업 본문 이후의 검색·평가 단계입니다. 원본 파싱·청킹 및 기존 실험 전체를 이관한 상태는 아닙니다. 원본 HWP/HWPX/PDF/Excel, 실제 코퍼스·정답·실버셋, 모델, 벡터, 실행 로그는 외부에 보관합니다. 예제는 합성 ID만 사용합니다.

python tools/check_publish.py는 Git index의 실제 blob에 대해 허용 경로, 금지 확장자, 바이너리·대용량 파일, 비밀키 패턴 및 개인 PC 경로를 검사합니다. 수동 코드 검토도 함께 수행해야 합니다.

선정 근거와 재현 검사 요약은 release/selection.json에 기록합니다. 원형은 로컬 통합 실험의 run.py, qwen_only_reranker.py입니다. 절대 경로와 정답 로딩을 추론 코드에서 제거하고 독립 평가기로 분리했습니다.

관련 프로젝트: [문서 파서](https://github.com/goodand/document-parser-mcp), [사람 검수 UI](https://github.com/goodand/decision-labeling-studio).
