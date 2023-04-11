from huggingface_hub import snapshot_download

model_path = "./model"
snapshot_download(repo_id="tuman/vit-rugpt2-image-captioning", local_dir=model_path)