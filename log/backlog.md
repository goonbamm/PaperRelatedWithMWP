
- [ ] [shell script 문법 공부](https://blog.d0ngd0nge.xyz/shell-script/)

- [ ] [data README 작성](/data/README.md)

- [ ] 모델 개선 아이디어
    + 1. KoBART 와 MWP 분야 모델 중 하나를 골라 영역별로 성능을 측정한다. 영역별로 앙상블을 취한다. type classifier 를 적용하면 되지 않을까. (https://huggingface.co/gogamza/kobart-base-v2), (https://huggingface.co/docs/transformers/model_doc/mbart#overview-of-mbart), (https://huggingface.co/sshleifer/tiny-mbart)

    + 2. mBERT_LSTM 의 아이디어처럼, 다중언어 데이터셋을 학습시켜서 성능 향상을 관찰한다. 이때, 영역별로도 관찰해본다. 그 이유는 언어가 다른 데이터셋의 유형 역시 중요할 수 있기 때문이다. 예컨대, 영어 중점 모델이지만 중국어 데이터가 규칙 찾기 관련 문항이 많다면, 규칙 찾기 관련 문항 정답률이 올랐는지 보는 것이다.
    
    + 3. STS 임베딩 같은 것을 활용할 순 없을까. 나중에 sentenceBERT 논문을 읽고 나서, 떠오르는 아이디어가 있다면 데이터 분포 등을 봐도 좋을 것 같다.

    + 4. OPERATOR, OPERAND 를 따로 생성하는 건 어떨까?!

    
