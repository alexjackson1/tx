FROM ghcr.io/nvidia/jax:nightly-2023-09-30

WORKDIR /usr/src/tx

COPY . .

RUN pip install -r requirements.txt

CMD ["pytest", "-v", "-s", "tests/"]