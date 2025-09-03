---
layout: default
title: What is Multimodal LLMs?
---

- What is multimodal?
  - Multimodal은 여러가지 데이터를 통합하여 더욱 정확한 AI모델을 구축하는 연구 분야임
  - Modality는 데이터를 얻는 방식을 의미함
  - 이미지, 텍스트, 음성, 비디오 등 여러종류의 데이터를 결합하여 활용하는 것임
 
- ViT: Vision Transformer
  - Transformer는 NLP분야에서 표준으로 자리잡았지만, CV분야에선 제한적으로 적용되었음
  - Attention을 CNN과 함께 사용하거나, CNN을 유지하며 일부를 Transformer로 대체하는 방식
  - 이미지를 패치로 분할 후 각 패치의 선형 임베딩을 Transformer의 입력으로 사용하는 방식
    (여기서 각각의 패치는 NLP에서의 토큰처럼 취급)
  - ViT는 대규모 데이터셋으로 학습이 되어야 Inductive Bias를 극복 할 수 있음
  - 관련 연구
    - Cordonnier et al. (2020)
      - 해당 연구는 2x2패치를 추출하여 저해상도 이미지만 가능하지만, ViT는 중간 해상도 이미지에도 적용이 가능함
    - Chen et al. (2020a)
      - 해당 연구는 해상도와 색상공간을 줄인 픽셀에 Transformer를 적용한 iGPT모델을 제안함
  - ViT 모델 구조
    1. 이미지 분할: 입력 이미지를 특정 패치로 분할
    2. 선형 임베딩: 각 패치를 벡터형태로 변환하여 고차원 공간에 매핑
    3. 위치 임베딩 추가: 각 패치의 위치정보를 유지하기 위해 추가함 (Transformer는 순서정보를 알 수 없기 때문)
    4. 인코더 입력: 선형 임베딩과 위치 임베딩을 결합하여 생성된 vector sequence를 인코더에 입력
    5. CLS토큰: CLS토큰을 sequence에 추가(분류작업을 위하여)
    6. 분류
  - [참고자료](https://taewan2002.medium.com/vit-vision-transformer-1d0555068f48)

- CLIP: Connecting text and images
  
