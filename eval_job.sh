# python eval.py \
#   --ckpt_path /work/hdd/bcsi/ndaithankar/playground/dense-ir-510/checkpoints/ebt-stepstep=012000.ckpt \
#   --encoder bert-base-uncased \
#   --collection_path /work/hdd/bcsi/ndaithankar/datasets/msmarco/collection.tsv \
#   --query_path /work/hdd/bcsi/ndaithankar/datasets/msmarco/queries.dev.tsv \
#   --qrels_path /work/hdd/bcsi/ndaithankar/datasets/msmarco/qrels.dev.tsv \
#   --top1000_path /work/hdd/bcsi/ndaithankar/datasets/msmarco/top1000.eval.tsv

python eval.py \
  --ckpt_path /work/hdd/bcsi/ndaithankar/playground/dense-ir-510/checkpoints/dense_ir_bert-base-uncased_ebm_margin_0.5_lr_0.0001_bs_4096_shared_encoder_False_trainable_encoder_False/last.ckpt \
  --encoder bert-base-uncased \
  --pairs_path /work/hdd/bcsi/ndaithankar/datasets/msmarco/top1000.dev.tsv \
  --qrels_path /work/hdd/bcsi/ndaithankar/datasets/msmarco/qrels.dev.tsv