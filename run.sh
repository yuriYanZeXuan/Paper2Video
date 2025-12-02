# 1. 设置环境变量 (根据您的 APIconn_test.py 逻辑)
export RUNWAY_API_KEY="9443f3be43594a3e81a3a9bc3e37c1dd"  # 或者 OPENAI_API_KEY
# export OPENAI_BASE_URL="https://your-azure-endpoint.com/openai" # 您的 Azure Endpoint
export RUNWAY_API_BASE="https://runway.devops.xiaohongshu.com/openai"
export RUNWAY_API_VERSION="2024-12-01-preview" # 或者您的 API 版本

# 2. 运行 Paper2Video (指定 model_name 为 azure-4o)
python src/pipeline.py \
  --model_name_t azure-4o \
  --model_name_v azure-4o \
  --paper_latex_root ./assets/demo/latex_proj \
  --ref_img ./assets/demo/zeyu.png \
  --ref_audio ./assets/demo/zeyu.wav