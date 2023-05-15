PRETRAIN=../mbart.cc25.v2 # fix if you moved the downloaded checkpoint
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

fairseq-train ./output/monolingual_En \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --arch mbart_base \
  --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang src \
  --target-lang tgt \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.2 \
  --optimizer adam \
  --adam-eps 1e-06 \
  --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay \
  --lr 3e-05 \
  --warmup-updates 2500 \
  --total-num-update 40000 \
  --dropout 0.3 \
  --attention-dropout 0.1 \
  --weight-decay 0.0 \
  --max-tokens 1024 \
  --update-freq 2 \
  --save-interval 1 \
  --save-interval-updates 5000 \
  --keep-interval-updates 10 \
  --no-epoch-checkpoints \
  --seed 222 \
  --log-format simple \
  --log-interval 2 \
  --restore-file $PRETRAIN \
  --reset-optimizer \
  --reset-meters \
  --reset-dataloader \
  --reset-lr-scheduler \
  --langs $langs \
  --ddp-backend legacy_ddp
