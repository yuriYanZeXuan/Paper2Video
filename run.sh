
# 2. 运行 Paper2Video (指定 model_name 为 azure-4o)
python src/pipeline.py \
  --model_name_t azure-4o \
  --model_name_v azure-4o \
  --paper_latex_root ./assets/demo/latex_proj \
  --ref_img ./assets/demo/zeyu.png \
  --ref_audio ./assets/demo/zeyu.wav