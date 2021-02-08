##### Install Cuda Without Root #####
https://stackoverflow.com/questions/39379792/install-cuda-without-root
##### Install Cudnn in Ubuntu #####
https://askubuntu.com/questions/1230645/when-is-cuda-gonna-be-released-for-ubuntu-20-04

```
export PATH=/home/guttikon/toolkit/bin/:$PATH
export LD_LIBRARY_PATH=/home/guttikon/toolkit/lib64
```

##### Note: To fix libcusolver.so.10 error #####
```
cd $LD_LIBRARY_PATH
sudo ln libcusolver.so.11 libcusolver.so.10  # hard link
```

##### Note: To fix ***** is not a symbolic link error #####
https://forums.developer.nvidia.com/t/tensorrt-4-installation-libcudnn-so-7-is-not-a-symbolic-link/60103
```
example: /sbin/ldconfig.real: /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8 is not a symbolic link
```

##### download google drive small file #####
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O FILENAME
```
##### download google drive large file #####
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt
```
