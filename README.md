python.exe -m venv DeepAIEnv
.\DeepAIEnv\Scripts\activate 

source DeepAIEnv/bin/activate 


python -m pip install --upgrade pip setuptools wheel

python3 -m pip install -r requirements.txt
python3 -m pip list

python -m pip -v config list
pip install pytest -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

C:\Users\Administrator\AppData\Roaming\pip\pip.ini
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com


#python -m pip install --upgrade tkinter


python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
$env:HTTP_PROXY="http://127.0.0.1:10809"
$env:HTTPS_PROXY="http://127.0.0.1:10809"