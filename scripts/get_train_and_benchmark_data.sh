# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

benchmark_data_folder="data/train_and_benchmark_data"
if [[ ! -d $benchmark_data_folder ]]; then
	mkdir -p $benchmark_data_folder
fi

fileid="1IDjXFnNnHf__MO5j_onw4YwR97oS8lAy"
filename="data/train_and_benchmark_data.zip"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}

unzip -d $benchmark_data_folder $filename
rm $filename

mv $benchmark_data_folder/data/* $benchmark_data_folder/
rm -r $benchmark_data_folder/data/
