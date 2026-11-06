# TIR Text Inline Reference

ComfyUI 프롬프트의 레퍼런스 호출을 찾아 이미지를 불러오고, **SnapRef** 방식으로 모델에 연결하는 노드 팩입니다. TIL과 같은 Sender/Receiver 흐름을 사용하며 Base 노드는 없습니다.

## 동작 방식과 샘플러

Sender는 프롬프트의 `<ref:이미지이름>` 호출을 `ref_stack`으로 전달하고, 출력 문자열에서는 인식한 호출 문법을 제거합니다. Receiver는 이미지 폴더에서 파일을 찾아 SnapRef 어댑터를 모델에 적용하고 `MODEL`을 출력합니다.

Receiver의 `MODEL`은 일반 **KSampler**에 연결할 수 있습니다. ComfyUI 패치 모델로 확산 모델을 실행하는 고급 샘플러에서도 사용할 수 있습니다. SnapRef의 기본 레퍼런스 적용은 diffusion model 안에서 처리됩니다. 프롬프트에 `g`를 명시해 추가 ref guidance를 켜면 SnapRef의 post-CFG 콜백도 사용합니다. 사용자 정의 Guider/샘플러가 ComfyUI post-CFG 콜백을 실행하지 않으면 그 추가 가이던스나 보정은 적용되지 않을 수 있습니다.

## 기존 모델 패치와 고급 샘플러에 연결하기

Receiver는 기존 모델 패치 뒤에 둡니다. 예를 들어 기존 워크플로의 마지막이 Block Compile이면 다음 순서로 연결합니다.

```text
모델 로드 → LoRA → 기존 모델 패치들 → Block Compile → TIR Ref Receiver
                                                        ├→ CFG Guider → SamplerCustomAdvanced
                                                        └→ Basic Scheduler ──────────┘
```

Receiver의 최종 `MODEL`을 CFG Guider와 Basic Scheduler 양쪽에 연결합니다. Basic Scheduler는 모델에서 스케줄 계산에 필요한 `model_sampling`을 읽고, 레퍼런스가 적용되는 모델 예측은 CFG Guider를 거쳐 실행됩니다. Block Compile 및 사용자 정의 샘플러와의 호환성은 각 노드의 구현과 버전에 따라 달라질 수 있습니다.

## 노드와 입력

- **TIR Ref - Sender**: 호출을 파싱해 `ref_stack`과 호출이 제거된 프롬프트를 출력합니다.
- **TIR Ref - Receiver**: 호출된 이미지를 폴더에서 찾아 선택한 SnapRef 어댑터로 모델을 패치합니다.
- **TIR Ref - Sender (Wireless)** / **TIR Ref - Receiver (Wireless)**: 채널 이름으로 같은 요청을 전달합니다.

Receiver에는 `MODEL`, `VAE`, `CLIP`, `ref_stack`, 이미지 폴더, SnapRef 어댑터, 하위 폴더 검색 설정이 있습니다. SnapRef에 프롬프트 임베딩을 전달하므로 CLIP 입력이 필요합니다. `include_subfolders`를 켜면 지정 폴더 아래의 하위 폴더까지 검색합니다.

## 프롬프트 문법

기본 문법은 다음과 같습니다.

```text
<ref:이미지이름>
```

| 부분 | 뜻 |
|---|---|
| `ref` | 레퍼런스 호출임을 나타내는 고정 키워드 |
| `이미지이름` | Receiver의 폴더에서 찾을 이미지 파일 이름. 확장자는 적어도 되고 생략해도 됩니다. |
| 옵션 | 필요하면 `:키=값` 형식으로 뒤에 추가합니다. |

확장자를 적으면 해당 파일을 찾고, 생략하면 같은 이름(stem)의 후보 중 경로순 첫 파일을 사용합니다. 하위 폴더는 Receiver의 `include_subfolders`가 켜진 경우에 검색합니다. 하위 폴더 안의 파일은 상대 경로와 확장자를 함께 적어 지정할 수 있습니다.

옵션은 필요할 때만 `키=값`으로 추가합니다.

| 키 | 뜻 | 범위 또는 값 |
|---|---|---|
| `g` | SnapRef의 추가 레퍼런스 가이던스. 명시하고 `1`이 아니면 활성화합니다. | 0–20 |
| `e` | 추가 가이던스 종료 지점 | 0–1 |
| `c` | 가이던스 보정: `0` = APG, `1` = rescale, `2` = none | 0, 1, 2 |
| `mp` | 레퍼런스 이미지 크기(메가픽셀) | 0.25–2 |
| `m` | SnapRef 모드: `0` = Character & Style, `1` = Character | 0, 1 |
| `w` | 이 호출 이미지의 상대 가중치. `1`은 학습 기본값입니다. | 0–2 |

기본값은 `m=0`, `w=1`, `e=1`, `c=0`, `mp=1`입니다. `g`를 생략하면 추가 레퍼런스 가이던스를 사용하지 않습니다.

```text
<ref:face:m=0:g=4.5:e=1:c=0:mp=1:w=1>
```

여러 이미지를 쓸 때는 호출을 반복합니다. 첫 호출의 `g`, `e`, `c`, `mp`, `m` 설정이 묶음 전체에 적용되고, 뒤 호출에서는 `w`를 이미지별로 다르게 지정할 수 있습니다.

```text
<ref:face:m=0><ref:fullbody:w=0.6>
```

## 이미지 검색

- 확장자를 쓰면 해당 파일을 선택합니다. 하위 폴더 검색 중 이름이 겹치면 경로순 첫 파일을 선택합니다.
- 확장자를 생략하면 동일한 파일 이름(stem)을 가진 이미지 중 경로순 첫 파일을 선택합니다.
- 여러 이미지는 SnapRef에 각각 독립된 레퍼런스로 전달됩니다.

## 의존성과 설치

ComfyUI에 [SnapRef](https://github.com/ArcAnima/SnapRef)가 설치되어 있어야 합니다. 어댑터 선택 목록은 SnapRef 노드가 제공하는 목록을 사용합니다.

`ComfyUI-TIR-Text-Inline-Reference` 폴더를 ComfyUI의 `custom_nodes` 폴더에 복사하고 ComfyUI를 재시작합니다.
