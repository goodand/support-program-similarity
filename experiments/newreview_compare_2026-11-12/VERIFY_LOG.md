# LLM 순위 실험 검증 기록

질의별 파일의 top_raw에서 순위를 다시 계산해 summary.json과 대조한다. 정가는 claude-api 스킬 단가표(2026-11-03) x 토큰, 캐시 할인 없이 계산.

## 2026-11-12T11:27:17+09:00 검증

| 조건 | 상태 | 가장 나쁜 순위 | 30위 밖 | 자기 번호 포함 | 모델 기록 | 재계산=요약 | CLI 비용 | 정가(캐시 없음) |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5_low_chunk | 통과 | 30위 밖 있음 | 5 | 27 | 0/35 | True | $45.41 | $0.57 |
| claude-haiku-5-5_low_name | 통과 | 18 | 0 | 0 | 0/35 | True | $6.18 | $0.09 |
| claude-opus-5-5_low_chunk | 통과 | 11 | 0 | 5 | 5/35 | True | $40.26 | $23.41 |
| claude-opus-5-5_low_name | 통과 | 19 | 0 | 8 | 2/35 | True | $6.38 | $3.41 |
| claude-sonnet-5-5_high_chunk | 미완료 (11/35) |  |  |  |  |  |  |  |
| claude-sonnet-5-5_high_name | 통과 | 19 | 0 | 0 | 35/35 | True | $3.45 | $1.78 |
| claude-sonnet-5-5_low_chunk | 통과 | 15 | 0 | 3 | 0/35 | True | $22.71 | $11.37 |
| claude-sonnet-5-5_low_name | 통과 | 18 | 0 | 1 | 0/35 | True | $2.71 | $1.37 |

## 2026-11-12T11:28:51+09:00 검증

| 조건 | 상태 | 가장 나쁜 순위 | 30위 밖 | 자기 번호 포함 | 모델 기록 | 재계산=요약 | CLI 비용 | 정가(캐시 없음) |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5_low_chunk | 통과 | 30위 밖 있음 | 5 | 27 | 0/35 | True | $45.41 | $0.57 |
| claude-haiku-5-5_low_name | 통과 | 18 | 0 | 0 | 0/35 | True | $6.18 | $0.09 |
| claude-opus-5-5_low_chunk | 통과 | 11 | 0 | 5 | 5/35 | True | $40.26 | $23.41 |
| claude-opus-5-5_low_name | 통과 | 19 | 0 | 8 | 2/35 | True | $6.38 | $3.41 |
| claude-sonnet-5-5_high_chunk | 미완료 (22/35) |  |  |  |  |  |  |  |
| claude-sonnet-5-5_high_name | 통과 | 19 | 0 | 0 | 35/35 | True | $3.45 | $1.78 |
| claude-sonnet-5-5_low_chunk | 통과 | 15 | 0 | 3 | 0/35 | True | $22.71 | $11.37 |
| claude-sonnet-5-5_low_name | 통과 | 18 | 0 | 1 | 0/35 | True | $2.71 | $1.37 |

## 2026-11-12T11:30:25+09:00 검증

| 조건 | 상태 | 가장 나쁜 순위 | 30위 밖 | 자기 번호 포함 | 모델 기록 | 재계산=요약 | CLI 비용 | 정가(캐시 없음) |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5_low_chunk | 통과 | 30위 밖 있음 | 5 | 27 | 0/35 | True | $45.41 | $0.57 |
| claude-haiku-5-5_low_name | 통과 | 18 | 0 | 0 | 0/35 | True | $6.18 | $0.09 |
| claude-opus-5-5_low_chunk | 통과 | 11 | 0 | 5 | 5/35 | True | $40.26 | $23.41 |
| claude-opus-5-5_low_name | 통과 | 19 | 0 | 8 | 2/35 | True | $6.38 | $3.41 |
| claude-sonnet-5-5_high_chunk | 통과 | 15 | 0 | 0 | 35/35 | True | $23.48 | $11.81 |
| claude-sonnet-5-5_high_name | 통과 | 19 | 0 | 0 | 35/35 | True | $3.45 | $1.78 |
| claude-sonnet-5-5_low_chunk | 통과 | 15 | 0 | 3 | 0/35 | True | $22.71 | $11.37 |
| claude-sonnet-5-5_low_name | 통과 | 18 | 0 | 1 | 0/35 | True | $2.71 | $1.37 |

## 2026-11-12T11:33:22+09:00 검증

| 조건 | 상태 | 가장 나쁜 순위 | 30위 밖 | 자기 번호 포함 | 모델 기록 | 재계산=요약 | CLI 비용 | 정가(캐시 없음) |
|---|---|---|---|---|---|---|---|---|
| claude-fable-5-2_low_name | 확인 필요 | 30위 밖 있음 | 35 | 0 | 0/35 | True | $0 | $0.0 |
  - q00.json: 응답 해석 실패
  - q01.json: 응답 해석 실패
  - q02.json: 응답 해석 실패
  - q03.json: 응답 해석 실패
  - q04.json: 응답 해석 실패
  - q05.json: 응답 해석 실패
  - q06.json: 응답 해석 실패
  - q07.json: 응답 해석 실패
  - q08.json: 응답 해석 실패
  - q09.json: 응답 해석 실패
  - q10.json: 응답 해석 실패
  - q11.json: 응답 해석 실패
  - q12.json: 응답 해석 실패
  - q13.json: 응답 해석 실패
  - q14.json: 응답 해석 실패
  - q15.json: 응답 해석 실패
  - q16.json: 응답 해석 실패
  - q17.json: 응답 해석 실패
  - q18.json: 응답 해석 실패
  - q19.json: 응답 해석 실패
  - q20.json: 응답 해석 실패
  - q21.json: 응답 해석 실패
  - q22.json: 응답 해석 실패
  - q23.json: 응답 해석 실패
  - q24.json: 응답 해석 실패
  - q25.json: 응답 해석 실패
  - q26.json: 응답 해석 실패
  - q27.json: 응답 해석 실패
  - q28.json: 응답 해석 실패
  - q29.json: 응답 해석 실패
  - q30.json: 응답 해석 실패
  - q31.json: 응답 해석 실패
  - q32.json: 응답 해석 실패
  - q33.json: 응답 해석 실패
  - q34.json: 응답 해석 실패
| claude-haiku-5-5_low_chunk | 통과 | 30위 밖 있음 | 5 | 27 | 0/35 | True | $45.41 | $0.57 |
| claude-haiku-5-5_low_name | 통과 | 18 | 0 | 0 | 0/35 | True | $6.18 | $0.09 |
| claude-opus-5-5_low_chunk | 통과 | 11 | 0 | 5 | 5/35 | True | $40.26 | $23.41 |
| claude-opus-5-5_low_name | 통과 | 19 | 0 | 8 | 2/35 | True | $6.38 | $3.41 |
| claude-sonnet-5-5_high_chunk | 통과 | 15 | 0 | 0 | 35/35 | True | $23.48 | $11.81 |
| claude-sonnet-5-5_high_name | 통과 | 19 | 0 | 0 | 35/35 | True | $3.45 | $1.78 |
| claude-sonnet-5-5_low_chunk | 통과 | 15 | 0 | 3 | 0/35 | True | $22.71 | $11.37 |
| claude-sonnet-5-5_low_name | 통과 | 18 | 0 | 1 | 0/35 | True | $2.71 | $1.37 |

## 2026-11-12T12:39:33+09:00 검증

| 조건 | 상태 | 가장 나쁜 순위 | 30위 밖 | 자기 번호 포함 | 모델 기록 | 재계산=요약 | CLI 비용 | 정가(캐시 없음) |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5_low_chunk | 통과 | 30위 밖 있음 | 5 | 27 | 0/35 | True | $45.41 | $0.57 |
| claude-haiku-5-5_low_name | 통과 | 18 | 0 | 0 | 0/35 | True | $6.18 | $0.09 |
| claude-opus-5-5_high_chunk | 확인 필요 | 30위 밖 있음 | 16 | 7 | 19/35 | True | $25.44 | $12.77 |
  - q19.json: 응답 해석 실패
  - q20.json: 응답 해석 실패
  - q21.json: 응답 해석 실패
  - q22.json: 응답 해석 실패
  - q23.json: 응답 해석 실패
  - q24.json: 응답 해석 실패
  - q25.json: 응답 해석 실패
  - q26.json: 응답 해석 실패
  - q27.json: 응답 해석 실패
  - q28.json: 응답 해석 실패
  - q29.json: 응답 해석 실패
  - q30.json: 응답 해석 실패
  - q31.json: 응답 해석 실패
  - q32.json: 응답 해석 실패
  - q33.json: 응답 해석 실패
  - q34.json: 응답 해석 실패
| claude-opus-5-5_high_name | 확인 필요 | 30위 밖 있음 | 2 | 7 | 35/35 | True | $6.84 | $3.5 |
  - q09.json: 응답 해석 실패
  - q29.json: 응답 해석 실패
| claude-opus-5-5_low_chunk | 통과 | 11 | 0 | 5 | 5/35 | True | $40.26 | $23.41 |
| claude-opus-5-5_low_name | 통과 | 19 | 0 | 8 | 2/35 | True | $6.38 | $3.41 |
| claude-sonnet-5-5_high_chunk | 통과 | 15 | 0 | 0 | 35/35 | True | $23.48 | $11.81 |
| claude-sonnet-5-5_high_name | 통과 | 19 | 0 | 0 | 35/35 | True | $3.45 | $1.78 |
| claude-sonnet-5-5_low_chunk | 통과 | 15 | 0 | 3 | 0/35 | True | $22.71 | $11.37 |
| claude-sonnet-5-5_low_name | 통과 | 18 | 0 | 1 | 0/35 | True | $2.71 | $1.37 |

## 2026-11-12T14:26:01+09:00 검증

| 조건 | 상태 | 가장 나쁜 순위 | 30위 밖 | 자기 번호 포함 | 모델 기록 | 재계산=요약 | CLI 비용 | 정가(캐시 없음) |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5_low_chunk | 통과 | 30위 밖 있음 | 5 | 27 | 0/35 | True | $45.41 | $0.57 |
| claude-haiku-5-5_low_name | 통과 | 18 | 0 | 0 | 0/35 | True | $6.18 | $0.09 |
| claude-opus-5-5_high_chunk | 통과 | 11 | 0 | 10 | 35/35 | True | $46.88 | $23.54 |
| claude-opus-5-5_high_name | 확인 필요 | 30위 밖 있음 | 2 | 7 | 35/35 | True | $6.84 | $3.5 |
  - q09.json: 응답 해석 실패
  - q29.json: 응답 해석 실패
| claude-opus-5-5_low_chunk | 통과 | 11 | 0 | 5 | 5/35 | True | $40.26 | $23.41 |
| claude-opus-5-5_low_name | 통과 | 19 | 0 | 8 | 2/35 | True | $6.38 | $3.41 |
| claude-sonnet-5-5_high_chunk | 통과 | 15 | 0 | 0 | 35/35 | True | $23.48 | $11.81 |
| claude-sonnet-5-5_high_name | 통과 | 19 | 0 | 0 | 35/35 | True | $3.45 | $1.78 |
| claude-sonnet-5-5_low_chunk | 통과 | 15 | 0 | 3 | 0/35 | True | $22.71 | $11.37 |
| claude-sonnet-5-5_low_name | 통과 | 18 | 0 | 1 | 0/35 | True | $2.71 | $1.37 |

## 2026-11-12T15:16:46+09:00 검증

| 조건 | 상태 | 가장 나쁜 순위 | 30위 밖 | 자기 번호 포함 | 모델 기록 | 재계산=요약 | CLI 비용 | 정가(캐시 없음) |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5_low_chunk | 통과 | 30위 밖 있음 | 5 | 27 | 0/35 | True | $45.41 | $0.57 |
| claude-haiku-5-5_low_name | 통과 | 18 | 0 | 0 | 0/35 | True | $6.18 | $0.09 |
| claude-opus-5-5_high_chunk | 통과 | 11 | 0 | 10 | 35/35 | True | $46.88 | $23.54 |
| claude-opus-5-5_high_name | 통과 | 16 | 0 | 7 | 35/35 | True | $6.84 | $3.5 |
| claude-opus-5-5_low_chunk | 통과 | 11 | 0 | 5 | 5/35 | True | $40.26 | $23.41 |
| claude-opus-5-5_low_name | 통과 | 19 | 0 | 8 | 2/35 | True | $6.38 | $3.41 |
| claude-sonnet-5-5_high_chunk | 통과 | 15 | 0 | 0 | 35/35 | True | $23.48 | $11.81 |
| claude-sonnet-5-5_high_name | 통과 | 19 | 0 | 0 | 35/35 | True | $3.45 | $1.78 |
| claude-sonnet-5-5_low_chunk | 통과 | 15 | 0 | 3 | 0/35 | True | $22.71 | $11.37 |
| claude-sonnet-5-5_low_name | 통과 | 18 | 0 | 1 | 0/35 | True | $2.71 | $1.37 |

## 2026-11-12T18:14:19+09:00 검증

| 조건 | 상태 | 가장 나쁜 순위 | 30위 밖 | 자기 번호 포함 | 모델 기록 | 재계산=요약 | CLI 비용 | 정가(캐시 없음) |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5_low_chunk | 통과 | 30위 밖 있음 | 5 | 27 | 0/35 | True | $45.41 | $0.57 |
| claude-haiku-5-5_low_name | 통과 | 18 | 0 | 0 | 0/35 | True | $6.18 | $0.09 |
| claude-opus-5-5_high_chunk | 통과 | 11 | 0 | 10 | 35/35 | True | $46.88 | $23.54 |
| claude-opus-5-5_high_name | 통과 | 16 | 0 | 7 | 35/35 | True | $6.84 | $3.5 |
| claude-opus-5-5_low_chunk | 통과 | 11 | 0 | 5 | 5/35 | True | $40.26 | $23.41 |
| claude-opus-5-5_low_name | 통과 | 19 | 0 | 8 | 2/35 | True | $6.38 | $3.41 |
| claude-sonnet-5-5_high_chunk | 통과 | 15 | 0 | 0 | 35/35 | True | $23.48 | $11.81 |
| claude-sonnet-5-5_high_name | 통과 | 19 | 0 | 0 | 35/35 | True | $3.45 | $1.78 |
| claude-sonnet-5-5_low_chunk | 통과 | 15 | 0 | 3 | 0/35 | True | $22.71 | $11.37 |
| claude-sonnet-5-5_low_name | 통과 | 18 | 0 | 1 | 0/35 | True | $2.71 | $1.37 |
