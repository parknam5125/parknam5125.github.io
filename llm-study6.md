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
  - 
