sudo apt-get install git-lfs
pip uninstall requests && pip install requests "httpx[socks]"

git clone https://github.com/svjack/cogvideox-factory

cd cogvideox-factory

pip install -r requirements.txt
pip install huggingface_hub accelerate
pip install git+https://github.com/huggingface/diffusers

huggingface-cli download \
  --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset \
  --local-dir video-dataset-disney

https://huggingface.co/THUDM/CogVideoX-5b-I2V

#### 2B
bash ./train_text_to_video_lora.sh

#### require memorys
bash ./train_image_to_video_lora.sh
