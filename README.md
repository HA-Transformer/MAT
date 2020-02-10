# HA-Transformer

The implementation of hybrid attentive Transformer (HAT).
The implementation is based on [fairseq](https://github.com/pytorch/fairseq).

## Model Training

### IWSLT14 De-En

1. Data preparation  
   See [prepare-iwslt14.sh](./examples/translation/prepare-iwslt14.sh).

2. Train HAT

    ```bash
    # To reproduce the 35.70 result in Table 1.
    N_a=3
    d=256
    d_h=2048
    rho=0.3

    python train.py /path/to/prepared/dataset \
        --seed 1 \
        --ddp-backend no_c10d \
        --task translation --arch transformer_mb_iwslt_de_en \
        --max-epoch 200 --max-update 10000000 \
        --share-all-embeddings \
        --optimizer adam --adam-betas '0.9 0.98' \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
        --clip-norm 0.0 --weight-decay 0.0 \
        --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
        --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
        --encoder-branches ${N_a} --decoder-branches ${N_a} --branch-dropout ${rho} \
        --encoder-pffn-branches 1 --decoder-pffn-branches 1 \
        --weight-decay 0.0 \
        --save-dir /path/to/saved/checkpoints \
        --max-tokens 4096 --update-freq 1 \
        --no-progress-bar --log-interval 25 --save-interval 1 --save-interval-updates 0 --keep-interval-updates 0 \
        --join-pffn --encoder-embed-dim ${d} --decoder-embed-dim ${d} --encoder-ffn-embed-dim ${d_h} --decoder-ffn-embed-dim ${d_h} \
        --relu-dropout 0.0 --attention-dropout 0.0 \
        > /path/to/log/file 2>&1
    ```

3. Train HAT + Proximal initialization

    ```bash
    # To reproduce the 36.22 result in Table 2.

    # 1. Train the corresponding baseline Transformer
    N_a=1
    d=256
    d_h=2048
    rho=0.0
    # Run the training command in previous section, change --save-dir to '/path/to/std/transformer'.

    # 2. Initialize parameters
    N_a=3
    python scripts/init-mbt-ckpt.py \
        /path/to/std/transformer/checkpoint_best.pt \
        /path/to/proximal-init-hat/checkpoint_last.pt \
        -N ${N_a} --NF 1 \
        --ro

    # 3. Train HAT
    N_a=3
    d=256
    d_h=2048
    rho=0.3
    # Run the training command in previous section, change --save-dir to '/path/to/proximal-init-hat'.
    ```

4. Inference HAT

    ```bash
    python generate.py /path/to/prepared/dataset \
        --gen-subset test \
        --path /path/to/saved/checkpoints/checkpoint_best.pt \
        --batch-size 128 \
        --beam 5 --lenpen 1.2 \
        --remove-bpe \
        --quiet --source-lang de --target-lang en \
        > /path/to/infer/log/file 2>&1
    ```
