# HA-Transformer

The implementation of multi-branch attentive Transformer (MAT).
The implementation is based on [fairseq](https://github.com/pytorch/fairseq).

## Model Training

### IWSLT14 De-En

1. Data preparation  
   1. See [prepare-iwslt14.sh](./examples/translation/prepare-iwslt14.sh) to preprocess the data.
   2. Binarize the dataset:

   ```bash
   # NOTE: Remember to add "--joined-dictionary" option.
   TEXT=examples/translation/iwslt14.tokenized.de-en
    python preprocess.py --source-lang de --target-lang en --trainpref ${TEXT}/train --validpref ${TEXT}/valid --testpref ${TEXT}/test --joined-dictionary --destdir /path/to/prepared/dataset
   ```

2. Train MAT

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
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
        --clip-norm 0.0 --weight-decay 0.0 \
        --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
        --dropout 0.3 --attention-dropout 0.0 --relu-dropout 0.0 \
        --encoder-branches ${N_a} --decoder-branches ${N_a} --branch-dropout ${rho} \
        --encoder-pffn-branches 1 --decoder-pffn-branches 1 \
        --weight-decay 0.0 \
        --save-dir /path/to/saved/checkpoints \
        --max-tokens 4096 --update-freq 1 \
        --no-progress-bar --log-interval 25 --save-interval 1 --save-interval-updates 0 --keep-interval-updates 0 \
        --join-pffn --encoder-embed-dim ${d} --decoder-embed-dim ${d} --encoder-ffn-embed-dim ${d_h} --decoder-ffn-embed-dim ${d_h} \
        > /path/to/log/file 2>&1
    ```

3. Train MAT + Proximal initialization

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
        /path/to/proximal-init-mat/checkpoint_last.pt \
        -N ${N_a} --NF 1 \
        --ro

    # 3. Train MAT
    N_a=3
    d=256
    d_h=2048
    rho=0.3
    # Run the training command in previous section, change --save-dir to '/path/to/proximal-init-mat'.
    ```

4. Inference MAT

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

### WMT14 En-De

1. Data preparation
   1. See [prepare-wmt14en2de.sh](./examples/translation/prepare-wmt14en2de.sh) to preprocess the data.
   2. Binarize the dataset:

   ```bash
   # NOTE: Remember to add "--joined-dictionary" option.
   TEXT=examples/translation/wmt14_en_de
   python preprocess.py --source-lang de --target-lang en --trainpref ${TEXT}/train --validpref ${TEXT}/valid --testpref ${TEXT}/test --joined-dictionary --destdir /path/to/prepared/dataset --nwordssrc 32768 --nwordstgt 32768
   ```

2. Train MAT

    ```bash
    N_a=2
    d=512
    d_h=12288
    rho=0.2

    # Train on 8 P40 GPUs; update_freq := 128 / num_gpus.
    python train.py /path/to/prepared/dataset \
        --seed 1 \
        --ddp-backend no_c10d --distributed-backend nccl --distributed-no-spawn \
        --task translation --arch transformer_mb_vaswani_wmt_en_de_big \
        --max-epoch 200 --max-update 10000000 \
        --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
        --clip-norm 0.0 --weight-decay 0.0 \
        --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
        --dropout 0.3 --attention-dropout 0.0 --relu-dropout 0.0 \
        --encoder-branches ${N_a} --decoder-branches ${N_a} --branch-dropout ${rho} \
        --encoder-pffn-branches 1 --decoder-pffn-branches 1 \
        --weight-decay 0.0 \
        --save-dir /path/to/saved/checkpoints \
        --max-tokens 4096 --update-freq 16 \
        --no-progress-bar --log-interval 25 --save-interval 1 --save-interval-updates 0 --keep-interval-updates 0 \
        --join-pffn --encoder-embed-dim ${d} --decoder-embed-dim ${d} --encoder-ffn-embed-dim ${d_h} --decoder-ffn-embed-dim ${d_h} \
        > /path/to/log/file 2>&1
    ```

3. Train MAT + Proximal initialization

    ```bash
    # To reproduce the 29.90 result in Table 4.

    # 1. Train the corresponding baseline Transformer
    N_a=1
    d=512
    d_h=12288
    rho=0.0
    # Run the training command in previous section, change --save-dir to '/path/to/std/transformer'.

    # 2. Initialize parameters
    N_a=2
    python scripts/init-mbt-ckpt.py \
        /path/to/std/transformer/checkpoint_best.pt \
        /path/to/proximal-init-mat/checkpoint_last.pt \
        -N ${N_a} --NF 1 \
        --ro

    # 3. Train MAT
    N_a=2
    d=512
    d_h=12288
    rho=0.2
    # Run the training command in previous section, change --save-dir to '/path/to/proximal-init-mat'.
    ```

4. Inference MAT

    ```bash
    python generate.py /path/to/prepared/dataset \
        --gen-subset test \
        --path /path/to/saved/checkpoints/checkpoint_best.pt \
        --batch-size 128 \
        --beam 4 --lenpen 0.6 \
        --remove-bpe \
        --quiet --source-lang de --target-lang en \
        > /path/to/infer/log/file 2>&1
    ```

### WMT19 En-De

1. Data preparation  
    We collect En-De translation data from [ACL 2019 Fourth Conference on Machine Translation (WMT19)](http://www.statmt.org/wmt19/translation-task.html), then follow [prepare-wmt14en2de.sh](./examples/translation/prepare-wmt14en2de.sh) to preprocess the data. **NOTE: We change the clean ratio to 2.0.**.

2. Train MAT

    ```bash
    N_a=2
    d=512
    d_h=12288
    rho=0.2

    # Train on 8 P40 GPUs; update_freq := 524288 / max_tokens / num_gpus.
    python train.py /path/to/prepared/dataset \
        --seed 1 \
        --ddp-backend no_c10d --distributed-backend nccl --distributed-no-spawn \
        --task translation --arch transformer_mb_vaswani_wmt_en_de_big \
        --max-update 800000 \
        --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
        --clip-norm 0.0 --weight-decay 0.0 \
        --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
        --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
        --encoder-layers 12 --decoder-layers 6 \
        --encoder-attention-heads 8 --decoder-attention-heads 8 \
        --encoder-branches ${N_a} --decoder-branches ${N_a} --branch-dropout ${rho} \
        --encoder-pffn-branches 1 --decoder-pffn-branches 1 \
        --weight-decay 0.0 \
        --save-dir /path/to/saved/checkpoints \
        --max-tokens 2048 --update-freq 32 \
        --no-progress-bar --log-interval 25 --save-interval 1 --save-interval-updates 1000 --keep-interval-updates 100 \
        --encoder-embed-dim ${d} --decoder-embed-dim ${d} --encoder-ffn-embed-dim ${d_h} --decoder-ffn-embed-dim ${d_h} \
        > /path/to/log/file 2>&1
    ```

3. Train MAT + Proximal initialization

    ```bash
    # To reproduce the 30.80 result in Table 7.

    # 1. Train the corresponding baseline Transformer
    N_a=1
    d=512
    d_h=12288
    rho=0.0
    # Run the training command in previous section, change --save-dir to '/path/to/std/transformer'.

    # 2. Initialize parameters
    N_a=2
    python scripts/init-mbt-ckpt.py \
        /path/to/std/transformer/checkpoint_best.pt \
        /path/to/proximal-init-mat/checkpoint_last.pt \
        -N ${N_a} --NF 1 \
        --ro

    # 3. Train MAT
    N_a=2
    d=512
    d_h=12288
    rho=0.2
    # Run the training command in previous section, change --save-dir to '/path/to/proximal-init-mat'.
    ```

4. Inference MAT

    ```bash
    python generate.py /path/to/prepared/dataset \
        --gen-subset test \
        --path /path/to/saved/checkpoints/checkpoint_best.pt \
        --batch-size 128 \
        --beam 4 --lenpen 0.6 \
        --remove-bpe \
        --quiet --source-lang de --target-lang en \
        > /path/to/infer/log/file 2>&1
    ```
