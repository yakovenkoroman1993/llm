from import_gpt2.gpt_download import download_and_load_gpt2

# url = (
#   "https://raw.githubusercontent.com/rasbt/"
#   "LLMs-from-scratch/main/ch05/"
#   "01_main-chapter-code/gpt_download.py"
# )
# filename = url.split('/')[-1]
# print("filename", filename)

# urllib.request.urlretrieve(url, filename)

settings, params = download_and_load_gpt2(
  model_size="124M", models_dir="gpt2"
)

print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())