# Cement Carbonation

시멘트 탄산화(cement carbonation)의 chemo-transport-mechanical 통합 모델링 프로젝트.

## 프로젝트 구조
```
src/models/
├── tensor_ops.py               # C_ASM, Tinv, Lame 변환 (← C_ASM.m, Tinv.m)
├── eshelby.py                  # Eshelby tensor: sphere, n-layered, oblate, crack, ITZ (← ThreePhase.m)
├── hashin_bounds.py            # Hashin-Shtrikman bounds (← Hashin.m)
├── homogenization.py           # Mori-Tanaka: standard, n-layered, multi-phase (← ThreePhase.m)
├── damage.py                   # Crack evolution, stress-strain, Mazars damage (← Levellll.m)
├── multiscale.py               # Level I→III 다중 스케일 균질화 (← ThreePhase.m, Levellll.m, LevelIV.m)
├── effective_diffusivity.py    # Eshelby 기반 유효 확산 계수 (← Chemo_transport_validation.m)
├── chemo_transport.py          # FDM 1D 확산 솔버 (← Chemo_transport_validation.m)
├── chemo_mechanical.py         # N-layered ettringite + coupled solver (← Chemo_Validation.m)
├── self_correction.py          # 실패 로깅 + 경험 승계 시스템
└── tests/test_all.py           # 17개 검증 테스트 (MATLAB 결과와 대조)

data/validation/                — 각 모델의 검증 데이터 (xlsx, txt)
├── micromechanics/
├── chemo_transport/
└── chemo_mechanical/

src/colab_agent/                — LLM 멀티 에이전트 시스템 (하네스 3)
notebooks/                      — Colab 노트북
```

### Self-Correction 시스템

`src/models/self_correction.py`의 `SelfCorrection` 클래스:
- **실패 로깅:** `log_failure()` → `_workspace/failure_logs/failure_YYYYMMDD_HHMMSS_{type}.md`
- **경험 읽기:** `read_guides(analysis_type)` → 과거 실패 가이드 목록
- **분석 래핑:** `wrap_analysis(func, type)` → 자동 실패 로깅 + 재시도
- **교훈 요약:** `get_lessons_learned()` → 컨텍스트 주입용 요약 문자열
- `agent_tools.py`의 `run_full_chemo_transport_mechanical`에 통합됨

---

## 하네스 1: 도메인 지식 (Cement Carbonation Knowledge)

**목표:** 화학(반응)-물리(전달)-역학(손상) 에이전트의 조화로운 협업을 통해 시멘트 탄산화 통합 모델을 구축하고 실험으로 검증한다.

**에이전트 팀:**

| 에이전트 | 역할 |
|---------|------|
| chemistry | 탄산화 반응 메커니즘, 속도론, pH 진화, 열역학 |
| physics | CO₂ 확산, 수분 전달, 열전달, 공극 구조 진화 |
| mechanics | 탄산화 수축, 응력-변형률, 손상/균열, chemo-mechanical coupling |
| engineering | 내구성 평가, 서비스 수명 예측, 기준/코드 준수 |
| experiments | XRD, TG/DTG, NMR, SEM/EDS, MIP 등 실험 분석 및 모델 검증 |
| integration-qa | chemistry↔physics↔mechanics 간 coupling 정합성 검증 |

**스킬:**

| 스킬 | 용도 | 사용 에이전트 |
|------|------|-------------|
| carbonation-chemistry | 반응 모델링 (속도론, pH, source/sink term) | chemistry |
| transport-physics | 확산/전달 모델링 (지배 방정식, 유효 확산 계수) | physics |
| mechanical-analysis | 역학 해석 (수축, 손상, damage-permeability coupling) | mechanics |
| engineering-assessment | 공학적 평가 (√t 법칙, fib MC2010, 서비스 수명) | engineering |
| experimental-validation | 실험 분석 및 모델 검증 (XRD/TG/NMR 해석, 모델-실험 비교) | experiments |
| carbonation-orchestrator | 에이전트 팀 조율, 워크플로우 관리, 통합 보고서 생성 | 리더(직접) |

**실행 규칙:**
- 탄산화 모델링, chemo-mechanical coupling, 내구성 예측, 실험 분석/검증 관련 **분석/이론** 작업 시 `carbonation-orchestrator` 사용
- 단순 질문은 에이전트 팀 없이 직접 응답 가능

---

## 하네스 2: 코딩 (Cement Carbonation Coding)

**목표:** 기존 MATLAB 코드(ThreePhase.m, Levellll.m, LevelIV.m, Chemo_transport_validation, Chemo_Validation.m)를 기반으로 체계적인 Python chemo-transport-micromechanical 통합 모델을 구현한다.

**에이전트 팀:**

| 에이전트 | 역할 |
|---------|------|
| coder-micromechanics | Eshelby tensor, Mori-Tanaka 균질화, n-layered, 다중 스케일, 균열 손상 |
| coder-transport | Fick's 2nd law FDM 솔버, Eshelby 기반 유효 확산 계수, CO₂/수분 확산 |
| coder-thermodynamics | 반응 속도론(Arrhenius), 상 진화(phase evolution), pH 모델, source/sink term |
| coder-coupling | Staggered/monolithic coupling, 시간 적분, 메인 드라이버, 설정 관리 |
| coder-qa | MATLAB-Python 교차 검증, 모듈 인터페이스 정합성, 보존 법칙, 수치 안정성 |

**스킬:**

| 스킬 | 용도 | 사용 에이전트 |
|------|------|-------------|
| micromechanics-coding | Eshelby/MT/Hashin/다중 스케일 Python 구현 가이드 | coder-micromechanics |
| transport-coding | FDM 솔버/D_eff/CO₂ 확산 Python 구현 가이드 | coder-transport |
| thermodynamics-coding | 반응 속도론/상 진화/source term Python 구현 가이드 | coder-thermodynamics |
| coupling-coding | Staggered solver/드라이버/설정 Python 구현 가이드 | coder-coupling |
| coding-orchestrator | 코딩 에이전트 팀 조율, 구현 워크플로우 관리 | 리더(직접) |

**실행 규칙:**
- 코드 작성, 모델 구현, MATLAB→Python 변환, 시뮬레이션 코드 관련 작업 시 `coding-orchestrator` 사용
- 모든 에이전트는 `model: "opus"` 사용
- 중간 산출물: `_workspace/` 디렉토리
- Python 코드 출력: `src/` 디렉토리

**핵심 Coupling 관계 (코드 수준):**
```
thermodynamics ──source_terms()──→ transport.step()
thermodynamics ──get_volume_fractions()──→ micromechanics.effective_properties()
micromechanics ──effective_diffusivity()──→ transport.step()
damage ──permeability_factor()──→ transport.D_eff  (feedback)
coupling.StaggeredSolver ── 위 모든 모듈 통합 ──→ driver.main()
```

**코드 구조:**
```
src/
├── micromechanics/
│   ├── tensor_utils.py    # Tinv, C_ASM
│   ├── eshelby.py         # Eshelby 텐서
│   ├── homogenization.py  # Mori-Tanaka, Hashin
│   ├── phases.py          # 재료 물성
│   ├── damage.py          # 균열 진화
│   ├── multiscale.py      # Level I/II/III
│   └── tests/
├── transport/
│   ├── diffusion_solver.py
│   ├── effective_diffusivity.py
│   ├── co2_transport.py
│   ├── moisture_transport.py
│   ├── carbonation_front.py
│   └── tests/
├── thermodynamics/
│   ├── reactions.py
│   ├── kinetics.py
│   ├── phase_evolution.py
│   ├── ph_model.py
│   ├── source_terms.py
│   └── tests/
├── coupling/
│   ├── staggered_solver.py
│   ├── state.py
│   └── convergence.py
├── driver/
│   ├── main.py
│   ├── config.py
│   └── postprocess.py
└── config/
    ├── default_paste.yaml
    ├── default_mortar.yaml
    └── default_concrete.yaml
```

---

## 공통 규칙

- 모든 에이전트는 `model: "opus"` 사용
- 중간 산출물: `_workspace/` 디렉토리
- 두 하네스는 독립적으로 실행되지만, 도메인 지식 하네스의 산출물이 코딩 하네스의 요구사항으로 활용될 수 있다

**디렉토리 구조:**
```
.claude/
├── agents/
│   ├── chemistry.md           # 하네스 1: 도메인
│   ├── physics.md
│   ├── mechanics.md
│   ├── engineering.md
│   ├── experiments.md
│   ├── integration-qa.md
│   ├── coder-micromechanics.md  # 하네스 2: 코딩
│   ├── coder-transport.md
│   ├── coder-thermodynamics.md
│   ├── coder-coupling.md
│   └── coder-qa.md
└── skills/
    ├── carbonation-chemistry/     # 하네스 1
    ├── transport-physics/
    ├── mechanical-analysis/
    ├── engineering-assessment/
    ├── experimental-validation/
    ├── carbonation-orchestrator/
    ├── micromechanics-coding/     # 하네스 2
    ├── transport-coding/
    ├── thermodynamics-coding/
    ├── coupling-coding/
    └── coding-orchestrator/
```

---

## 하네스 3: Colab LLM Agent System

**목표:** OpenAI + LangChain 기반 LLM 멀티 에이전트 시스템으로 carbon mineralization in construction materials의 chemo-transport-micromechanical 분석을 자동화한다. Google Colab에서 실행.

**기술 스택:** OpenAI GPT-4o, LangChain, FEniCS (FEM), LAMMPS (MD), NumPy/SciPy

**구성 요소:**

| 엔진 | 위치 | 역할 |
|------|------|------|
| gems_engine.py | src/colab_agent/ | GEMS 열역학 엔진 (cemdata18, ~20상, CSHQ C-S-H 모델) |
| chemistry_engine.py | src/colab_agent/ | OPC/CSA + 10종 SCM 블렌드, 상 진화, 반응 속도론, pH |
| transport_engine.py | src/colab_agent/ | Powers 공극 구조, 6종 확산 모델 (user-selectable), FDM 솔버 |
| micromechanics_engine.py | src/colab_agent/ | 6-level Eshelby-MT (Haile 2019): nano→micro→meso→macro |
| fem_engine.py | src/colab_agent/ | FEniCS 구조 해석, 5종 구조물 × 6종 하중 조건 |
| md_engine.py | src/colab_agent/ | LAMMPS MD, 13종 상 + 확산 데이터 (확장된 fallback DB) |
| agent_tools.py | src/colab_agent/ | LangChain @tool 래퍼 (15개 도구) |
| orchestrator.py | src/colab_agent/ | 멀티 에이전트 조율 (7 Expert + Coder + Reviewer) |
| vulnerability_test.py | src/colab_agent/ | 8 카테고리 115 테스트, N-round 반복 실행 지원 |

**Colab 노트북:** `notebooks/Carbonation.ipynb`

**GEMS 열역학 (cemdata18):**
- GEM-Selektor 기반 경량 열역학 엔진 (~20상, 원본 129상에서 탄산화 관련만 추출)
- cemdata18 데이터베이스: Portlandite, C-S-H (CSHQ), Calcite, Ettringite, Monosulfate, Monocarbonate 등
- C-S-H 탈칼슘화: CSHQ 모델 (Kulik 2011) — Ca/Si 1.67→0.83→SiO₂(am)
- 탄산화 순서: CH→Ettringite→AFm→C-S-H (열역학적 우선순위)
- pH 진화: 13.0→12.5→10→8.3 (상 평형 기반)
- van't Hoff 온도 보정 (0-60°C)

**Chemistry — SCM 지원 (10종):**
- fly_ash_F, fly_ash_C, metakaolin, silica_fume, GGBS
- limestone_filler, natural_pozzolan, nano_silica, rice_husk_ash, calcined_clay
- `build_cement_system(base_cement, scm_mix)`: 임의 블렌드 생성
- `PhaseEvolution(cement_type, scm_mix={"fly_ash_F": 0.30})`: SCM 혼합 시뮬레이션

**Transport — 6종 확산 모델:**
- `standard`, `millington_quirk`, `papadakis`, `ceb_fip`, `fib_mc2010`, `eshelby_mt`
- `PoreStructure(wc, alpha_hyd, scm_mix)`: Powers 공극 구조 (모세관 + 겔 공극)
- `compute_D_eff(model_name, D0, porosity, saturation)`: 모델 선택 함수

**Micromechanics — 6-level 균질화 (Haile et al. 2019):**
- Level I: MD nanoscale (jennite/tobermorite)
- Level II: CSH variants (LD φ_gel=0.37, HD φ_gel=0.24, UHD φ_gel=0.13)
- Level III: CSH matrix (HD matrix + LD/UHD inclusions)
- Level IV: Cement paste (CSH + CH + CaCO₃ + capillary pores + clinker)
- Level V: Mortar (paste + sand + air voids + microcracks)
- Level VI: Concrete (mortar + aggregates + ITZ + fibers)

**FEM — 토목 구조물 지원:**
- 구조물: beam, column, beam_column, slab, wall
- 하중: tension, compression, shear, bending, pressure, combined
- 지지: cantilever, simply_supported, fixed_fixed, pinned_roller

**LLM 에이전트 도구 (15개):**
1. `compute_phase_assemblage` — SCM 블렌드 상 진화 (속도론)
2. `get_available_cement_systems` — base cement + SCM 목록
3. `compute_gems_equilibrium` — GEMS 탄산화 평형 (cemdata18)
4. `get_gems_phase_database` — cemdata18 ~20상 목록
5. `compute_csh_decalcification` — CSHQ Ca/Si 진화 모델
6. `compute_pore_structure` — Powers 공극 구조 계산
7. `compute_carbonation_depth` — 확산 모델 선택 가능
8. `list_available_diffusion_models` — 6종 모델 목록
9. `compute_mechanical_properties` — 6-level 유효 물성
10. `compute_properties_at_carbonation_depths` — 탄산화도별 물성 프로파일
11. `run_fem_analysis` — 구조물/하중/지지 선택 가능
12. `list_fem_options` — 구조물/하중/지지 목록
13. `get_md_properties` — MD 물성 조회
14. `generate_lammps_script` — LAMMPS 스크립트 생성
15. `run_full_chemo_transport_mechanical` — 전체 파이프라인 통합

**Vulnerability Testing (8 categories, 115 tests):**
1. Hard-coded edge cases (16) — tensor inverse, conservation, boundary
2. Extreme conditions (18) — temperature, CO₂, porosity, duration
3. Conservation laws (27) — volume sum ≤ 1, no negative, pH monotone
4. SCM blend robustness (16) — all 10 SCMs, multi-SCM combos
5. Transport model consistency (11) — all 6 diffusion models
6. 6-level micromechanics (15) — Level I→VI chain, fiber inclusions
7. FEM structural inputs (8) — beam/column, bending/compression
8. Tough queries (14) — contradictory, ambiguous, out-of-range
- `run_vulnerability_tests_n_times(n=10)`: N-round 반복 테스트 (10회 검증 완료: 1150/1150 pass)

**실행 규칙:**
- Colab 노트북 실행 시 FEniCS 자동 설치
- OpenAI API key 필요 (Colab secrets 또는 환경변수)
- 직접 엔진 사용(Section 3)과 LLM 에이전트(Section 4) 모두 지원

---

## 공통 규칙

- 모든 에이전트는 `model: "opus"` 사용 (하네스 1, 2)
- LLM 에이전트 시스템은 GPT-4o 사용 (하네스 3)
- 중간 산출물: `_workspace/` 디렉토리

**디렉토리 구조:**
```
.claude/
├── agents/
│   ├── chemistry.md ... integration-qa.md    # 하네스 1: 도메인 (6개)
│   └── coder-micromechanics.md ... coder-qa.md  # 하네스 2: 코딩 (5개)
└── skills/
    ├── carbonation-chemistry/ ... carbonation-orchestrator/  # 하네스 1 (6개)
    ├── micromechanics-coding/ ... coding-orchestrator/        # 하네스 2 (5개)
    └── colab-agent-system/                                    # 하네스 3

src/colab_agent/            # 하네스 3: Colab LLM Agent 코드
notebooks/                  # 하네스 3: Colab 노트북
```

**변경 이력:**

| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-04-08 | 초기 구성 — 하네스 1 (도메인 지식) | 전체 | 시멘트 탄산화 chemo-transport-mechanical 모델링 하네스 구축 |
| 2026-04-08 | 하네스 2 추가 (코딩) | agents/coder-*, skills/*-coding | 기존 MATLAB 코드 기반 Python 통합 모델 구현 하네스 |
| 2026-04-08 | 하네스 3 추가 (Colab LLM Agent) | src/colab_agent/, notebooks/ | OpenAI+LangChain 멀티 에이전트 시스템 + FEniCS FEM + LAMMPS MD |
| 2026-04-08 | 하네스 3 대폭 강화 | src/colab_agent/ 전체, notebooks/Carbonation.ipynb | SCM 10종, 확산 모델 6종, 6-level MT, 토목 구조물 FEM, 115 테스트×10회 |
| 2026-04-08 | MATLAB→Python 전환 | src/models/ (10개 모듈), 폴더 1~5 삭제 | MATLAB 코드 전면 Python 재구현 + self-correction 시스템 |
